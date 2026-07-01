#!/usr/bin/env python3
"""Debug Conv2D - simplified version."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S

# Very small test
BATCH = 1
IN_CHANNELS = 16
OUT_CHANNELS = 16
IN_H = 16
IN_W = 16
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 14
OUT_W = IN_W - KERNEL_W + 1  # 14
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 144

WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
frag_size = MFMA_K // 4

gemm_m = BATCH * OUT_H * OUT_W  # 196


@substrate.jit
def mfma_accum(A, B, C):
    lane = S.thread_id(0)
    c_lane = C[lane]
    m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
    C[lane] = c_lane


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
    conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)

    expected = conv(x)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {conv.weight.shape}")
    print(f"Output shape: {expected.shape}")
    print(f"gemm_m: {gemm_m}, GEMM_K: {GEMM_K}")
    print()

    # Build A matrix (im2col)
    x_f16 = x.to(torch.float16)
    w_f16 = conv.weight.to(torch.float16)

    A_gemm = torch.zeros((gemm_m, GEMM_K), dtype=torch.float16, device=device)
    for hw_idx in range(gemm_m):
        batch = hw_idx // (OUT_H * OUT_W)
        hw_in_batch = hw_idx % (OUT_H * OUT_W)
        oh = hw_in_batch // OUT_W
        ow = hw_in_batch % OUT_W
        for k in range(GEMM_K):
            c = k // KERNEL_AREA
            spatial = k % KERNEL_AREA
            kh = spatial // KERNEL_W
            kw = spatial % KERNEL_W
            ih = oh + kh
            iw = ow + kw
            A_gemm[hw_idx, k] = x_f16[batch, c, ih, iw]

    # Build B matrix
    B_gemm = w_f16.reshape(OUT_CHANNELS, GEMM_K).t().contiguous()

    print(f"A_gemm shape: {A_gemm.shape}")
    print(f"B_gemm shape: {B_gemm.shape}")

    # Reference GEMM
    C_ref = torch.matmul(A_gemm.float(), B_gemm.float())
    print(f"C_ref shape: {C_ref.shape}")

    # Compare with expected
    expected_gemm = expected.permute(0, 2, 3, 1).reshape(gemm_m, OUT_CHANNELS)
    diff_ref = torch.abs(C_ref - expected_gemm)
    print(f"C_ref vs expected: max diff = {diff_ref.max().item():.6f}")

    # Test direct simple GEMM approach
    print("\n" + "=" * 60)
    print("Test: Direct simple GEMM (no tile iteration)")
    print("=" * 60)

    # Create simple matrices
    A_simple = A_gemm[0:16, :]
    B_simple = B_gemm[:, 0:16]
    expected_simple = torch.matmul(A_simple.float(), B_simple.float())

    print(f"A_simple shape: {A_simple.shape}")
    print(f"B_simple shape: {B_simple.shape}")
    print(f"expected_simple sum: {expected_simple.sum().item():.4f}")

    k_tiles = GEMM_K // MFMA_K
    print(f"K tiles: {k_tiles}")

    C_simple = torch.zeros((16, 16), dtype=torch.float32, device=device)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    for k_tile in range(k_tiles):
        k_start = k_tile * MFMA_K

        A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            i = lane % 16
            k_block = lane // 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                A_shuffled[lane, t] = A_simple[i, k_start + k]

        B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            k_block = lane // 16
            j = lane % 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                B_shuffled[lane, t] = B_simple[k_start + k, j]

        A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

        mfma_accum[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_simple[4 * g + t, j] = C_shuffled[lane, t]

    diff_simple = torch.abs(C_simple - expected_simple)
    print(f"Simple GEMM sum: {C_simple.sum().item():.4f}")
    print(f"Max diff: {diff_simple.max().item():.6f}")

    if diff_simple.max().item() < 0.01:
        print("\n✓ Simple GEMM works!")
    else:
        print("\n✗ Simple GEMM failed")
