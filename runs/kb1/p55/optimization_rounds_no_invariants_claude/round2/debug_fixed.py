#!/usr/bin/env python3
"""Test Conv2D MFMA with corrected accumulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Conv2D dimensions
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

gemm_m = BATCH * OUT_H * OUT_W  # 289

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Test: Fixed Conv2D MFMA")
print("=" * 60)
print(f"gemm_m: {gemm_m}, GEMM_K: {GEMM_K}")
print(f"Spatial tiles: {(gemm_m + MFMA_M - 1) // MFMA_M}")
print(f"K tiles: {GEMM_K // MFMA_K}")
print()

torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)
w = conv.weight

expected = conv(x)

# Convert to f16
x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

# Build A matrix (im2col)
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

# Reference GEMM
C_ref = torch.matmul(A_gemm.float(), B_gemm.float())

# Full implementation with CORRECTED accumulation
spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N
k_tiles = GEMM_K // MFMA_K

y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

for spatial_tile in range(spatial_tiles):
    for oc_tile in range(oc_tiles):
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        # Iterate over K tiles and ACCUMULATE into C_shuffled
        for k_tile in range(k_tiles):
            k_start = k_tile * MFMA_K

            # Shuffle A
            A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
            for lane in range(WARP_SIZE):
                i = lane % MFMA_M
                k_block = lane // MFMA_M

                for t in range(frag_size):
                    k = k_block * frag_size + t
                    k_global = k_start + k

                    row = spatial_tile * MFMA_M + i
                    if row < gemm_m and k_global < GEMM_K:
                        A_shuffled[lane, t] = A_gemm[row, k_global]

            # Shuffle B
            B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
            for lane in range(WARP_SIZE):
                k_block = lane // MFMA_N
                j = lane % MFMA_N

                for t in range(frag_size):
                    k = k_block * frag_size + t
                    k_global = k_start + k

                    col = oc_tile * MFMA_N + j
                    if k_global < GEMM_K and col < OUT_CHANNELS:
                        B_shuffled[lane, t] = B_gemm[k_global, col]

            A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
            B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

            # This accumulates into C_shuffled!
            gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        # After all K tiles, unshuffle and write to output
        for lane in range(WARP_SIZE):
            g = lane // 16
            j = lane % 16

            for t in range(4):
                row = g * 4 + t
                col = j

                hw_idx = spatial_tile * MFMA_M + row
                oc = oc_tile * MFMA_N + col

                if hw_idx < gemm_m and oc < OUT_CHANNELS:
                    y_gemm[hw_idx, oc] = C_shuffled[lane, t]  # = is correct here, all K tiles accumulated in C_shuffled

output_mfma = y_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_mfma = torch.abs(output_mfma - expected)
print(f"Fixed MFMA: max diff = {diff_mfma.max().item():.6f}")
print(f"Result matches: {torch.allclose(output_mfma, expected, rtol=1e-2, atol=0.1)}")

# Compare with torch GEMM
diff_vs_gemm = torch.abs(y_gemm - C_ref)
print(f"MFMA vs torch GEMM: max diff = {diff_vs_gemm.max().item():.6f}")

print("\n" + "=" * 60)
print("Test: Split-K with corrected logic")
print("=" * 60)

SPLIT_K_SLICES = 2
C_PER_SPLIT = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES  # 16

y_gemm_split = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

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
            C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

            for k_start in range(k_start_global, k_end_global, MFMA_K):
                A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
                for lane in range(WARP_SIZE):
                    i = lane % MFMA_M
                    k_block = lane // MFMA_M

                    for t in range(frag_size):
                        k = k_block * frag_size + t
                        k_global = k_start + k

                        row = spatial_tile * MFMA_M + i
                        if row < gemm_m and k_global < GEMM_K:
                            A_shuffled[lane, t] = A_gemm[row, k_global]

                B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
                for lane in range(WARP_SIZE):
                    k_block = lane // MFMA_N
                    j = lane % MFMA_N

                    for t in range(frag_size):
                        k = k_block * frag_size + t
                        k_global = k_start + k

                        col = oc_tile * MFMA_N + j
                        if k_global < GEMM_K and col < OUT_CHANNELS:
                            B_shuffled[lane, t] = B_gemm[k_global, col]

                A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
                B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

                gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

            # Accumulate into y_gemm across splits
            for lane in range(WARP_SIZE):
                g = lane // 16
                j = lane % 16

                for t in range(4):
                    row = g * 4 + t
                    col = j

                    hw_idx = spatial_tile * MFMA_M + row
                    oc = oc_tile * MFMA_N + col

                    if hw_idx < gemm_m and oc < OUT_CHANNELS:
                        y_gemm_split[hw_idx, oc] += C_shuffled[lane, t]  # += for split-K

output_split = y_gemm_split.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_split = torch.abs(output_split - expected)
print(f"Split-K MFMA: max diff = {diff_split.max().item():.6f}")
print(f"Result matches: {torch.allclose(output_split, expected, rtol=1e-2, atol=0.1)}")
