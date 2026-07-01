#!/usr/bin/env python3
"""Debug MFMA tile alignment and boundary conditions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Test with perfectly aligned dimensions first
BATCH = 1
IN_CHANNELS = 32
OUT_CHANNELS = 16
IN_H = 16
IN_W = 16
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 14
OUT_W = IN_W - KERNEL_W + 1  # 14
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 288

# MFMA parameters
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32

gemm_m = BATCH * OUT_H * OUT_W  # 1 * 14 * 14 = 196

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Tile alignment analysis")
print("=" * 60)
print(f"gemm_m = {gemm_m}")
print(f"GEMM_K = {GEMM_K}")
print(f"MFMA_M = {MFMA_M}")
print(f"MFMA_N = {MFMA_N}")
print(f"MFMA_K = {MFMA_K}")
print()

# Check alignment
spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N
k_tiles = (GEMM_K + MFMA_K - 1) // MFMA_K

print(f"Spatial tiles: {spatial_tiles} (gemm_m / MFMA_M = {gemm_m} / {MFMA_M} = {gemm_m / MFMA_M})")
print(f"OC tiles: {oc_tiles} (OUT_CHANNELS / MFMA_N = {OUT_CHANNELS} / {MFMA_N} = {OUT_CHANNELS / MFMA_N})")
print(f"K tiles: {k_tiles} (GEMM_K / MFMA_K = {GEMM_K} / {MFMA_K} = {GEMM_K / MFMA_K})")
print()

# Check if dimensions are aligned
print(f"gemm_m aligned: {gemm_m % MFMA_M == 0}")
print(f"OUT_CHANNELS aligned: {OUT_CHANNELS % MFMA_N == 0}")
print(f"GEMM_K aligned: {GEMM_K % MFMA_K == 0}")
print()

# Create input and weights
torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)
w = conv.weight

# Expected output
expected = conv(x)

# Convert to f16
x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

# Build A matrix (im2col)
print("Building A matrix...")
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
print(f"Reference GEMM vs Conv2D: max diff = {diff_ref.max().item():.6f}")

# Test MFMA with correct dimensions
print("\n" + "=" * 60)
print("Testing MFMA with tile handling")
print("=" * 60)

# Initialize output
y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

# Shuffle functions
def shuffle_a_correct(A_gemm, spatial_start, k_start, gemm_m, GEMM_K):
    """Shuffle A matrix fragment for MFMA with correct boundary handling."""
    frag_size = MFMA_K // 4  # 8 f16 per lane

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        i = lane % MFMA_M  # row index (0-15)
        k_block = lane // MFMA_M  # K block (0-3)

        for t in range(frag_size):
            k = k_block * frag_size + t  # local K index (0-31)
            k_global = k_start + k

            row = spatial_start + i

            if row < gemm_m and k_global < GEMM_K:
                A_shuffled[lane, t] = A_gemm[row, k_global]
            # else: keep as zero

    return A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

def shuffle_b_correct(B_gemm, oc_tile, k_start, OUT_CHANNELS, GEMM_K):
    """Shuffle B matrix fragment for MFMA with correct boundary handling."""
    frag_size = MFMA_K // 4

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        k_block = lane // MFMA_N  # K block (0-3)
        j = lane % MFMA_N  # column index (0-15)

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            col = oc_tile * MFMA_N + j

            if k_global < GEMM_K and col < OUT_CHANNELS:
                B_shuffled[lane, t] = B_gemm[k_global, col]
            # else: keep as zero

    return B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

def unshuffle_c_correct(C_shuffled, spatial_start, oc_tile, y_gemm, gemm_m, OUT_CHANNELS):
    """Unshuffle C fragment to GEMM output with correct boundary handling."""
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

# Process all tiles
for spatial_tile in range(spatial_tiles):
    for oc_tile in range(oc_tiles):
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        for k_tile in range(k_tiles):
            k_start = k_tile * MFMA_K

            A_packed = shuffle_a_correct(A_gemm, spatial_tile * MFMA_M, k_start, gemm_m, GEMM_K)
            B_packed = shuffle_b_correct(B_gemm, oc_tile, k_start, OUT_CHANNELS, GEMM_K)

            gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        unshuffle_c_correct(C_shuffled, spatial_tile * MFMA_M, oc_tile, y_gemm, gemm_m, OUT_CHANNELS)

output_mfma = y_gemm.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_mfma = torch.abs(output_mfma - expected)
print(f"MFMA result: max diff = {diff_mfma.max().item():.6f}")

# Compare with direct torch GEMM on same data
diff_vs_gemm = torch.abs(y_gemm - C_ref)
print(f"MFMA vs torch GEMM: max diff = {diff_vs_gemm.max().item():.6f}")

# Debug: check a single tile
print("\n" + "=" * 60)
print("Debug single tile")
print("=" * 60)

# Test just the first spatial tile and first OC tile
spatial_tile = 0
oc_tile = 0

C_shuffled_debug = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

for k_tile in range(k_tiles):
    k_start = k_tile * MFMA_K

    A_packed = shuffle_a_correct(A_gemm, spatial_tile * MFMA_M, k_start, gemm_m, GEMM_K)
    B_packed = shuffle_b_correct(B_gemm, oc_tile, k_start, OUT_CHANNELS, GEMM_K)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled_debug)

# Expected for this tile
expected_tile = C_ref[0:16, 0:16]  # First 16 rows, first 16 columns

# Unshuffle and compare
actual_tile = torch.zeros((16, 16), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        row = g * 4 + t
        actual_tile[row, j] = C_shuffled_debug[lane, t]

diff_tile = torch.abs(actual_tile - expected_tile)
print(f"First tile max diff: {diff_tile.max().item():.6f}")
print(f"First tile mean diff: {diff_tile.mean().item():.6f}")

# Print some values
print(f"\nExpected tile[0,0]: {expected_tile[0,0].item():.4f}")
print(f"Actual tile[0,0]: {actual_tile[0,0].item():.4f}")
print(f"\nExpected tile[0,:5]: {expected_tile[0,:5]}")
print(f"Actual tile[0,:5]: {actual_tile[0,:5]}")
