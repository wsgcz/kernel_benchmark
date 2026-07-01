#!/usr/bin/env python3
"""Test the MFMA Conv2D model with non-accumulating kernel and external accumulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Small test parameters
BATCH = 1
IN_CHANNELS = 32
OUT_CHANNELS = 16
IN_H = 19
IN_W = 19
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1
OUT_W = IN_W - KERNEL_W + 1
KERNEL_AREA = KERNEL_H * KERNEL_W
GEMM_K = IN_CHANNELS * KERNEL_AREA

WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
frag_size = MFMA_K // 4

SPLIT_K_SLICES = 2
C_PER_SPLIT = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES

gemm_m = BATCH * OUT_H * OUT_W


def shuffle_a(x_nchw, spatial_start, k_start, device):
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


def shuffle_b(w_oihw, oc_tile, k_start, device):
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"gemm_m: {gemm_m}, GEMM_K: {GEMM_K}")

    torch.manual_seed(42)
    x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
    conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)

    expected = conv(x)

    x_f16 = x.to(torch.float16)
    w_f16 = conv.weight.to(torch.float16)

    y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

    spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
    oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N

    print(f"Spatial tiles: {spatial_tiles}, OC tiles: {oc_tiles}")

    for split_k_id in range(SPLIT_K_SLICES):
        c_start = split_k_id * C_PER_SPLIT
        c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)

        if c_start >= IN_CHANNELS:
            continue

        k_start_global = c_start * KERNEL_AREA
        k_end_global = c_end * KERNEL_AREA

        print(f"Split {split_k_id}: K = {k_start_global} to {k_end_global - 1}")

        for spatial_tile in range(spatial_tiles):
            for oc_tile in range(oc_tiles):
                # Accumulate K tiles into a local result
                C_local = torch.zeros((16, 16), dtype=torch.float32, device=device)

                for k_start in range(k_start_global, k_end_global, MFMA_K):
                    A_packed = shuffle_a(x_f16, spatial_tile * MFMA_M, k_start, device)
                    B_packed = shuffle_b(w_f16, oc_tile, k_start, device)
                    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

                    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

                    # Accumulate into C_local
                    for lane in range(WARP_SIZE):
                        g = lane // 16
                        j = lane % 16
                        for t in range(4):
                            C_local[4 * g + t, j] += C_shuffled[lane, t]

                # Write to y_gemm with boundary checks
                for i in range(16):
                    for j in range(16):
                        hw_idx = spatial_tile * MFMA_M + i
                        oc = oc_tile * MFMA_N + j
                        if hw_idx < gemm_m and oc < OUT_CHANNELS:
                            y_gemm[hw_idx, oc] += C_local[i, j]

    output = y_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()

    diff = torch.abs(output - expected)
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    if torch.allclose(output, expected, rtol=1e-2, atol=0.1):
        print("\n✓ TEST PASSED!")
    else:
        print("\n✗ TEST FAILED")
