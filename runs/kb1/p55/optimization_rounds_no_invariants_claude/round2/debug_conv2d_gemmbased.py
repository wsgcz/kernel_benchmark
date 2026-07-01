#!/usr/bin/env python3
"""Test split-K Conv2D with MFMA using GEMM formulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Conv2D dimensions (smaller for testing)
BATCH = 2
IN_CHANNELS = 32
OUT_CHANNELS = 32
IN_H = 8
IN_W = 8
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 6
OUT_W = IN_W - KERNEL_W + 1  # 6
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 32 * 9 = 288

# MFMA parameters
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32  # Two MFMA calls per K tile

# Split-K parameters
SPLIT_K_SLICES = 2
C_PER_SPLIT = (IN_CHANNELS + SPLIT_K_SLICES - 1) // SPLIT_K_SLICES  # 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("Conv2D Configuration")
print("=" * 60)
print(f"BATCH: {BATCH}")
print(f"IN_CHANNELS: {IN_CHANNELS}")
print(f"OUT_CHANNELS: {OUT_CHANNELS}")
print(f"IN_H x IN_W: {IN_H} x {IN_W}")
print(f"OUT_H x OUT_W: {OUT_H} x {OUT_W}")
print(f"KERNEL: {KERNEL_H} x {KERNEL_W}")
print(f"GEMM_K: {GEMM_K}")
print(f"SPLIT_K_SLICES: {SPLIT_K_SLICES}")
print(f"C_PER_SPLIT: {C_PER_SPLIT}")
print(f"K tiles total: {GEMM_K // MFMA_K}")
print()

# K tile analysis
print("K tile analysis:")
for split_id in range(SPLIT_K_SLICES):
    c_start = split_id * C_PER_SPLIT
    c_end = min(IN_CHANNELS, c_start + C_PER_SPLIT)
    k_start_global = c_start * KERNEL_AREA
    k_end_global = c_end * KERNEL_AREA
    k_tiles_in_split = (k_end_global - k_start_global) // MFMA_K
    print(f"Split {split_id}: channels {c_start}-{c_end-1}, K={k_start_global}-{k_end_global-1}, {k_tiles_in_split} K tiles")

print("\n" + "=" * 60)
print("Testing GEMM-based Conv2D")
print("=" * 60)

# Create input and weights
torch.manual_seed(42)
x = torch.randn((BATCH, IN_CHANNELS, IN_H, IN_W), dtype=torch.float32, device=device)
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_H, bias=False).to(device)
w = conv.weight  # (OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W)

# Expected output
expected = conv(x)
print(f"Expected output shape: {expected.shape}")

# Convert to f16
x_f16 = x.to(torch.float16)
w_f16 = w.to(torch.float16)

# GEMM formulation: im2col
# A: (batch * out_h * out_w) x (in_channels * kernel_h * kernel_w)
# B: (in_channels * kernel_h * kernel_w) x out_channels
# C: (batch * out_h * out_w) x out_channels

gemm_m = BATCH * OUT_H * OUT_W

# Build A matrix (im2col)
print("\nBuilding A matrix (im2col)...")
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

# Build B matrix (weights reshaped)
print("Building B matrix...")
B_gemm = w_f16.reshape(OUT_CHANNELS, GEMM_K).t().contiguous()  # (GEMM_K, OUT_CHANNELS)

# Compute GEMM using torch.matmul (reference)
print("Computing reference GEMM...")
C_ref = torch.matmul(A_gemm.float(), B_gemm.float())
output_ref = C_ref.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()

diff_ref = torch.abs(output_ref - expected)
print(f"Reference GEMM vs PyTorch Conv2D: max diff = {diff_ref.max().item():.6f}")

# Now test MFMA-based computation
print("\n" + "=" * 60)
print("Testing MFMA-based GEMM")
print("=" * 60)

# Initialize output
y_gemm = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

# Process tiles
spatial_tiles = (gemm_m + MFMA_M - 1) // MFMA_M
oc_tiles = (OUT_CHANNELS + MFMA_N - 1) // MFMA_N
k_tiles = (GEMM_K + MFMA_K - 1) // MFMA_K

print(f"Spatial tiles: {spatial_tiles}")
print(f"Output channel tiles: {oc_tiles}")
print(f"K tiles: {k_tiles}")

# Shuffle and unshuffle functions
def shuffle_a(A_gemm, spatial_start, k_start):
    """Shuffle A matrix fragment for MFMA."""
    frag_size = MFMA_K // 4  # 8 f16 per lane

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        i = lane % MFMA_M  # row index
        k_block = lane // MFMA_M  # K block

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            row = spatial_start + i
            if row < gemm_m and k_global < GEMM_K:
                A_shuffled[lane, t] = A_gemm[row, k_global]

    return A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

def shuffle_b(B_gemm, oc_tile, k_start):
    """Shuffle B matrix fragment for MFMA."""
    frag_size = MFMA_K // 4

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)

    for lane in range(WARP_SIZE):
        k_block = lane // MFMA_N
        j = lane % MFMA_N  # column index

        for t in range(frag_size):
            k = k_block * frag_size + t
            k_global = k_start + k

            col = oc_tile * MFMA_N + j
            if k_global < GEMM_K and col < OUT_CHANNELS:
                B_shuffled[lane, t] = B_gemm[k_global, col]

    return B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

def unshuffle_c(C_shuffled, spatial_start, oc_tile, y_gemm):
    """Unshuffle C fragment to GEMM output."""
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

# Test without split-K first
print("\nTest 1: No split-K")
y_gemm_nosplit = torch.zeros((gemm_m, OUT_CHANNELS), dtype=torch.float32, device=device)

for spatial_tile in range(spatial_tiles):
    for oc_tile in range(oc_tiles):
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        for k_tile in range(k_tiles):
            k_start = k_tile * MFMA_K

            A_packed = shuffle_a(A_gemm, spatial_tile * MFMA_M, k_start)
            B_packed = shuffle_b(B_gemm, oc_tile, k_start)

            gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        unshuffle_c(C_shuffled, spatial_tile * MFMA_M, oc_tile, y_gemm_nosplit)

output_nosplit = y_gemm_nosplit.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_nosplit = torch.abs(output_nosplit - expected)
print(f"No split-K: max diff = {diff_nosplit.max().item():.6f}")

# Test with split-K
print("\nTest 2: With split-K")
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
                A_packed = shuffle_a(A_gemm, spatial_tile * MFMA_M, k_start)
                B_packed = shuffle_b(B_gemm, oc_tile, k_start)

                gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

            unshuffle_c(C_shuffled, spatial_tile * MFMA_M, oc_tile, y_gemm_split)

output_split = y_gemm_split.reshape(BATCH, OUT_H, OUT_W, OUT_CHANNELS).permute(0, 3, 1, 2).contiguous()
diff_split = torch.abs(output_split - expected)
print(f"Split-K: max diff = {diff_split.max().item():.6f}")
print(f"Result matches: {torch.allclose(output_split, expected, rtol=1e-2, atol=0.1)}")
