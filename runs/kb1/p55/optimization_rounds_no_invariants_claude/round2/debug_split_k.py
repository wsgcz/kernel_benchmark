#!/usr/bin/env python3
"""Debug script to trace split-K logic."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shape constants
BATCH = 8
IN_CHANNELS = 64
OUT_CHANNELS = 128
IN_H = 512
IN_W = 1024
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 510
OUT_W = IN_W - KERNEL_W + 1  # 1022
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 576
MFMA_K = 32

# Test split-K parameters
SPLIT_K_SLICES = 2
C_PER_SPLIT = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES  # 32

print("=" * 60)
print("Split-K Configuration Debug")
print("=" * 60)
print(f"IN_CHANNELS: {IN_CHANNELS}")
print(f"KERNEL_AREA: {KERNEL_AREA}")
print(f"GEMM_K: {GEMM_K}")
print(f"MFMA_K: {MFMA_K}")
print(f"SPLIT_K_SLICES: {SPLIT_K_SLICES}")
print(f"C_PER_SPLIT: {C_PER_SPLIT}")
print()

# Compute K tiles for each split
all_k_tiles = []
for split_k_id in range(SPLIT_K_SLICES):
    c_start = split_k_id * C_PER_SPLIT
    c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)

    if c_start >= IN_CHANNELS:
        continue

    k_start_global = c_start * KERNEL_AREA
    k_end_global = c_end * KERNEL_AREA

    split_k_tiles = list(range(k_start_global, k_end_global, MFMA_K))
    all_k_tiles.extend(split_k_tiles)

    print(f"Split {split_k_id}:")
    print(f"  Channels: {c_start} to {c_end-1}")
    print(f"  K range: {k_start_global} to {k_end_global-1}")
    print(f"  K tiles: {split_k_tiles}")
    print(f"  Total K tiles in this split: {len(split_k_tiles)}")
    print()

# Check for missing or duplicate K tiles
all_k_tiles_sorted = sorted(all_k_tiles)
print(f"All K tiles (sorted): {all_k_tiles_sorted}")
print(f"Total K tiles: {len(all_k_tiles_sorted)}")
print()

# Expected K tiles
expected_k_tiles = list(range(0, GEMM_K, MFMA_K))
print(f"Expected K tiles: {expected_k_tiles}")
print(f"Expected total: {len(expected_k_tiles)}")
print()

# Check coverage
missing = set(expected_k_tiles) - set(all_k_tiles_sorted)
duplicates = [k for k in all_k_tiles_sorted if all_k_tiles_sorted.count(k) > 1]
print(f"Missing K tiles: {missing}")
print(f"Duplicate K tiles: {set(duplicates)}")
print()

# Check K range coverage
print("K range coverage check:")
for split_k_id in range(SPLIT_K_SLICES):
    c_start = split_k_id * C_PER_SPLIT
    c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)

    if c_start >= IN_CHANNELS:
        continue

    k_start_global = c_start * KERNEL_AREA
    k_end_global = c_end * KERNEL_AREA

    # Verify channel ranges
    print(f"Split {split_k_id} covers channels {c_start} to {c_end-1}")

    # For each K tile, verify it maps to correct channels
    for k_start in range(k_start_global, k_end_global, MFMA_K):
        k_end = min(k_start + MFMA_K, k_end_global)
        c_start_tile = k_start // KERNEL_AREA
        c_end_tile = (k_end - 1) // KERNEL_AREA
        print(f"  K tile {k_start}-{k_end-1}: channels {c_start_tile} to {c_end_tile}")

print()
print("=" * 60)
print("Testing actual computation")
print("=" * 60)

# Create small test case
torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device='cuda')
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).cuda()
w = conv.weight

# Expected output
expected = conv(x)

# Now test the GEMM approach without split-K (just to verify)
x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

# GEMM formulation: im2col style
# A: (batch * out_h * out_w) x (in_channels * kernel_h * kernel_w)
# B: (in_channels * kernel_h * kernel_w) x out_channels
gemm_m = BATCH * OUT_H * OUT_W
A_gemm = torch.zeros((gemm_m, GEMM_K), dtype=torch.float16, device='cuda')
B_gemm = w_f16.reshape(OUT_CHANNELS, GEMM_K).t().contiguous()

# Fill A matrix
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

# Compute GEMM
C_gemm = torch.matmul(A_gemm.float(), B_gemm.float())

# Convert to NCHW
actual_gemm = C_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()

diff = torch.abs(actual_gemm - expected.float())
print(f"GEMM approach (no split-K): max diff = {diff.max().item():.6f}")

# Now test with split-K accumulation
C_split = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device='cuda')

for split_k_id in range(SPLIT_K_SLICES):
    c_start = split_k_id * C_PER_SPLIT
    c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)

    if c_start >= IN_CHANNELS:
        continue

    k_start_global = c_start * KERNEL_AREA
    k_end_global = c_end * KERNEL_AREA

    # Extract A slice for this split
    k_slice_size = k_end_global - k_start_global
    A_slice = A_gemm[:, k_start_global:k_end_global]
    B_slice = B_gemm[k_start_global:k_end_global, :]

    # Compute partial GEMM
    C_partial = torch.matmul(A_slice.float(), B_slice.float())

    # Accumulate
    C_split += C_partial

    print(f"Split {split_k_id}: computed partial GEMM for K={k_start_global} to {k_end_global-1}")

# Convert to NCHW
actual_split = C_split.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()

diff_split = torch.abs(actual_split - expected.float())
print(f"Split-K GEMM approach: max diff = {diff_split.max().item():.6f}")

print()
print("Done!")
