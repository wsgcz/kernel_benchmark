#!/usr/bin/env python3
"""Test the MFMA Conv2D model with smaller shapes to verify correctness."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import substrate
import substrate.language as S

# Override constants for small test
BATCH = 1
IN_CHANNELS = 32
OUT_CHANNELS = 16
IN_H = 19
IN_W = 19
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 17
OUT_W = IN_W - KERNEL_W + 1  # 17
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 288

# MFMA parameters
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
frag_size = MFMA_K // 4  # 8

# Split-K parameters
SPLIT_K_SLICES = 2
C_PER_SPLIT = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES

gemm_m = BATCH * OUT_H * OUT_W


@substrate.jit
def mfma_16x16x16_f16_kernel(
    A: S.Tensor((64, 4), S.u32),
    B: S.Tensor((64, 4), S.u32),
    C: S.Tensor((64, 4), S.f32),
):
    """Single MFMA operation: C += A @ B"""
    lane = S.thread_id(0)
    c_lane = C[lane]
    m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
    C[lane] = c_lane


def _shuffle_a_fragment(x_nchw, spatial_start, k_start, device):
    A_shuffled = torch.zeros((WARP_SIZE, 8), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        i = lane % MFMA_M
        k_block = lane // MFMA_M

        hw_idx = spatial_start + i
        batch = hw_idx // (OUT_H * OUT_W)
        hw_in_batch = hw_idx % (OUT_H * OUT_W)
        oh = hw_in_batch // OUT_W
        ow = hw_in_batch % OUT_W

        for t in range(8):
            k = k_block * 8 + t
            k_global = k_start + k

            if k_global >= GEMM_K:
                continue

            c = k_global // KERNEL_AREA
            spatial = k_global % KERNEL_AREA
            kh = spatial // KERNEL_W
            kw = spatial % KERNEL_W

            ih = oh + kh
            iw = ow + kw

            if batch < BATCH and c < IN_CHANNELS and ih < IN_H and iw < IN_W:
                A_shuffled[lane, t] = x_nchw[batch, c, ih, iw]

    return A_shuffled.view(torch.int32).view(WARP_SIZE, 4)


def _shuffle_b_fragment(w_oihw, oc_tile, k_start, device):
    B_shuffled = torch.zeros((WARP_SIZE, 8), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        k_block = lane // MFMA_N
        j = lane % MFMA_N

        oc = oc_tile * MFMA_N + j

        for t in range(8):
            k = k_block * 8 + t
            k_global = k_start + k

            if k_global >= GEMM_K:
                continue

            c = k_global // KERNEL_AREA
            spatial = k_global % KERNEL_AREA
            kh = spatial // KERNEL_W
            kw = spatial % KERNEL_W

            if oc < OUT_CHANNELS and c < IN_CHANNELS:
                B_shuffled[lane, t] = w_oihw[oc, c, kh, kw]

    return B_shuffled.view(torch.int32).view(WARP_SIZE, 4)


def _unshuffle_c_to_gemm(C_shuffled, spatial_start, oc_tile, y_gemm, device):
    for lane in range(WARP_SIZE):
        g = lane // MFMA_N
        j = lane % MFMA_N

        for t in range(4):
            row = g * 4 + t
            col = j

            hw_idx = spatial_start + row
            oc = oc_tile * MFMA_N + col

            if hw_idx < gemm_m and oc < OUT_CHANNELS:
                y_gemm[hw_idx, oc] += C_shuffled[lane, t]


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ModelNew, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        device = x.device

        x_f16 = x.to(torch.float16)
        x_nchw = x_f16.contiguous()

        w_f16 = self.conv2d.weight.to(device=device, dtype=torch.float16)
        w_oihw = w_f16.contiguous()

        y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

        spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
        oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N

        for split_k_id in range(SPLIT_K_SLICES):
            c_start = split_k_id * C_PER_SPLIT
            c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)

            if c_start >= IN_CHANNELS:
                continue

            k_start_global = c_start * KERNEL_AREA
            k_end_global = c_end * KERNEL_AREA

            for spatial_tile in range(spatial_tiles):
                for oc_tile in range(oc_tiles):
                    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

                    for k_start in range(k_start_global, k_end_global, MFMA_K):
                        A_packed = _shuffle_a_fragment(x_nchw, spatial_tile * MFMA_M, k_start, device)
                        B_packed = _shuffle_b_fragment(w_oihw, oc_tile, k_start, device)

                        mfma_16x16x16_f16_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](
                            A_packed, B_packed, C_shuffled
                        )

                    _unshuffle_c_to_gemm(C_shuffled, spatial_tile * MFMA_M, oc_tile, y_gemm, device)

        y = y_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
        return y.to(torch.float32)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"gemm_m: {gemm_m}, GEMM_K: {GEMM_K}")
    print(f"Spatial tiles: {(gemm_m + MFMA_M - 1) // MFMA_M}")
    print(f"K tiles: {GEMM_K // MFMA_K}")
    print()

    torch.manual_seed(42)
    x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
    model = ModelNew(IN_CHANNELS, OUT_CHANNELS, 3).to(device)

    print("Running MFMA Conv2D...")
    with torch.no_grad():
        actual = model(x)

    print("Running PyTorch Conv2D...")
    w = model.conv2d.weight.to(torch.float32)
    expected = F.conv2d(x, w, bias=None)

    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual_cpu - expected_cpu)).item()
    mean_diff = torch.mean(torch.abs(actual_cpu - expected_cpu)).item()

    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    if torch.allclose(actual_cpu, expected_cpu, rtol=1e-2, atol=0.1):
        print("\n✓ TEST PASSED: Results match within tolerance!")
    else:
        print("\n✗ TEST FAILED: Results do not match")
        print(f"actual[0,0,0,0:5] = {actual_cpu[0,0,0,0:5]}")
        print(f"expected[0,0,0,0:5] = {expected_cpu[0,0,0,0:5]}")
