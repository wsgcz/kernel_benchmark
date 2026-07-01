#!/usr/bin/env python3
"""Compare accumulating kernel vs non-accumulating kernel for multi-tile GEMM."""
import torch
import substrate
import substrate.language as S
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16 as gemm_no_accum  # Overwrites

WARP_SIZE = 64

@substrate.jit
def gemm_accum(
    A: S.Tensor((64, 4), S.u32),
    B: S.Tensor((64, 4), S.u32),
    C: S.Tensor((64, 4), S.f32),
):
    """Accumulating kernel: C += A @ B"""
    lane = S.thread_id(0)
    c_lane = C[lane]  # Load current value
    m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
    C[lane] = c_lane


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frag_size = 8
    MFMA_K = 32

    # Test with K=64 (two K tiles)
    M, N, K = 16, 16, 64
    torch.manual_seed(42)

    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    expected = torch.matmul(A.float(), B.float())

    print("Test: Multi-tile GEMM with K=64 (2 tiles)")
    print(f"Expected sum: {expected.sum().item():.4f}")
    print()

    # Method 1: Non-accumulating kernel with += accumulation
    print("Method 1: Non-accumulating kernel + += accumulation")
    C_result1 = torch.zeros((M, N), dtype=torch.float32, device=device)

    for k_tile in range(K // MFMA_K):
        k_start = k_tile * MFMA_K

        # Shuffle A
        A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            i = lane % 16
            k_block = lane // 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                A_shuffled[lane, t] = A[i, k_start + k]

        # Shuffle B
        B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            k_block = lane // 16
            j = lane % 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                B_shuffled[lane, t] = B[k_start + k, j]

        A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        gemm_no_accum[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        # Accumulate
        for lane in range(WARP_SIZE):
            g = lane // 16
            j = lane % 16
            for t in range(4):
                C_result1[4 * g + t, j] += C_shuffled[lane, t]

    diff1 = torch.abs(C_result1 - expected)
    print(f"  Max diff: {diff1.max().item():.6f}")
    print(f"  Sum: {C_result1.sum().item():.4f}")

    # Method 2: Accumulating kernel (same C_shuffled for all K tiles)
    print("\nMethod 2: Accumulating kernel (reuse C_shuffled)")
    C_result2 = torch.zeros((M, N), dtype=torch.float32, device=device)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    for k_tile in range(K // MFMA_K):
        k_start = k_tile * MFMA_K

        # Shuffle A
        A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            i = lane % 16
            k_block = lane // 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                A_shuffled[lane, t] = A[i, k_start + k]

        # Shuffle B
        B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            k_block = lane // 16
            j = lane % 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                B_shuffled[lane, t] = B[k_start + k, j]

        A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

        gemm_accum[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unshuffle
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_result2[4 * g + t, j] = C_shuffled[lane, t]

    diff2 = torch.abs(C_result2 - expected)
    print(f"  Max diff: {diff2.max().item():.6f}")
    print(f"  Sum: {C_result2.sum().item():.4f}")

    print()
    if diff2.max().item() < 0.01:
        print("✓ Accumulating kernel works correctly!")
    else:
        print("✗ Accumulating kernel has issues")
        print(f"C_shuffled after 2 tiles: {C_shuffled[0, :].tolist()}")
