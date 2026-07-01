#!/usr/bin/env python3
"""Debug im2col transformation and Conv2D tile iteration."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Simple Conv2D test
BATCH = 1
IN_CHANNELS = 32
OUT_CHANNELS = 16
IN_H = 19  # So OUT_H = 17 = 16 + 1 for proper alignment
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

gemm_m = BATCH * OUT_H * OUT_W  # 289

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Configuration")
print("=" * 60)
print(f"BATCH: {BATCH}, IN_CHANNELS: {IN_CHANNELS}, OUT_CHANNELS: {OUT_CHANNELS}")
print(f"IN_H x IN_W: {IN_H} x {IN_W}")
print(f"OUT_H x OUT_W: {OUT_H} x {OUT_W}")
print(f"GEMM_K: {GEMM_K}")
print(f"gemm_m: {gemm_m}")
print(f"Spatial tiles: {(gemm_m + MFMA_M - 1) // MFMA_M}")
print(f"OC tiles: {(OUT_CHANNELS + MFMA_N - 1) // MFMA_N}")
print(f"K tiles: {GEMM_K // MFMA_K}")
print()

torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)
w = conv.weight

# Expected output
expected = conv(x)

# Convert to f16
x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

print("=" * 60)
print("Test 1: Verify im2col is correct")
print("=" * 60)

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
output_ref = C_ref.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()

diff_ref = torch.abs(output_ref - expected)
print(f"im2col + torch GEMM vs Conv2D: max diff = {diff_ref.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 2: MFMA on single tile")
print("=" * 60)

# Test just the first spatial tile (rows 0-15) and first OC tile (cols 0-15)
spatial_tile = 0
oc_tile = 0

C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)
frag_size = MFMA_K // 4  # 8

# Process all K tiles for this output tile
for k_tile in range(GEMM_K // MFMA_K):
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

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

# Unshuffle
C_tile = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        row = g * 4 + t
        C_tile[row, j] = C_shuffled[lane, t]

# Expected for this tile
expected_tile = C_ref[0:16, 0:16]

diff_tile = torch.abs(C_tile - expected_tile)
print(f"Single tile MFMA: max diff = {diff_tile.max().item():.6f}")

print("\n" + "=" * 60)
print("Test 3: Full MFMA with proper tile iteration")
print("=" * 60)

# Full implementation
spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N
k_tiles = GEMM_K // MFMA_K

y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

for spatial_tile in range(spatial_tiles):
    for oc_tile in range(oc_tiles):
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

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

            gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        # Unshuffle and store
        for lane in range(WARP_SIZE):
            g = lane // 16
            j = lane % 16

            for t in range(4):
                row = g * 4 + t
                col = j

                hw_idx = spatial_tile * MFMA_M + row
                oc = oc_tile * MFMA_N + col

                if hw_idx < gemm_m and oc < OUT_CHANNELS:
                    y_gemm[hw_idx, oc] = C_shuffled[lane, t]  # Use = instead of += for no split-K

output_mfma = y_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_mfma = torch.abs(output_mfma - expected)
print(f"Full MFMA: max diff = {diff_mfma.max().item():.6f}")

# Compare with torch GEMM result
diff_vs_gemm = torch.abs(y_gemm - C_ref)
print(f"MFMA vs torch GEMM: max diff = {diff_vs_gemm.max().item():.6f}")
