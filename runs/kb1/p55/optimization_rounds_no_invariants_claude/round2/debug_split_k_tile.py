#!/usr/bin/env python3
"""Debug split-K with proper K tile handling."""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Test dimensions
M, N, K = 16, 16, 32  # Total K=32
WARP_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Create GEMM matrices
torch.manual_seed(0)
A = torch.randn((M, K), dtype=torch.float16, device=device)
B = torch.randn((K, N), dtype=torch.float16, device=device)

# Expected result
expected = torch.matmul(A.float(), B.float())
print(f"Expected result shape: {expected.shape}")
print(f"Expected[0,0] = {expected[0,0].item():.4f}")

# Test 1: Single kernel (K=32) with two MFMA calls
print("\n" + "=" * 60)
print("Test 1: Single kernel (K=32) with two MFMA calls")
print("=" * 60)

frag_size = 8  # 8 f16 = 4 u32 per lane for K=32

A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    i = lane % 16
    k_block = lane // 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        A_shuffled[lane, t] = A[i, k]

B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
for lane in range(WARP_SIZE):
    k_block = lane // 16
    j = lane % 16
    for t in range(frag_size):
        k = k_block * frag_size + t
        B_shuffled[lane, t] = B[k, j]

A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

C1 = torch.zeros((M, N), dtype=torch.float32, device=device)
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C1[4 * g + t, j] = C_shuffled[lane, t]

diff1 = torch.abs(C1 - expected)
print(f"Single kernel: max diff = {diff1.max().item():.6f}")
print(f"C1[0,0] = {C1[0,0].item():.4f}")

# Test 2: Split-K with K tiles of 32 each
# With K=32 and SPLIT_K=2, each split has K_split = 16
# But MFMA needs K=32 per call! So we need to handle this differently.

print("\n" + "=" * 60)
print("Test 2: Split-K analysis")
print("=" * 60)

SPLIT_K = 2
K_per_split = K // SPLIT_K  # = 16

print(f"Total K = {K}")
print(f"SPLIT_K = {SPLIT_K}")
print(f"K_per_split = {K_per_split}")
print(f"MFMA processes K=16 per call (mfma_16x16x16)")
print()

# With K_per_split = 16, we can use a single mfma_16x16x16 call per split
# But the kernel is written to make TWO MFMA calls!
# We need to either:
# 1. Modify the kernel to make only one MFMA call when K_per_split = 16
# 2. Or adjust the fragment packing

# Let me try option 2: pack two splits into one MFMA call
# For split 0: A[:, 0:16], B[0:16, :]
# For split 1: A[:, 16:32], B[16:32, :]

# The standard kernel packs:
# m_a[0] = first 4 f16, m_a[1] = next 4 f16
# These correspond to K = 0:16 and K = 16:32

# So I can run the full kernel once for the full K=32, OR
# I can run two separate passes and accumulate.

# Let me try the two-pass approach:

print("Testing two-pass split-K approach...")

C2 = torch.zeros((M, N), dtype=torch.float32, device=device)

# Process each split
for split_id in range(SPLIT_K):
    k_start = split_id * K_per_split
    k_end = k_start + K_per_split

    # Extract slices
    A_slice = A[:, k_start:k_end]  # (16, 16)
    B_slice = B[k_start:k_end, :]  # (16, 16)

    # For K=16, frag_size = 16 // 4 = 4
    frag_size_split = K_per_split // 4  # = 4 f16 = 2 u32

    print(f"Split {split_id}: k_start={k_start}, k_end={k_end}")
    print(f"  A_slice shape: {A_slice.shape}")
    print(f"  B_slice shape: {B_slice.shape}")
    print(f"  frag_size_split: {frag_size_split}")

    # Shuffle for K=16
    # With K=16, we have k_block = lane // 16, but this gives k_block = 0 for all lanes 0-15
    # And k_block = 1 for lanes 16-31, etc.
    # But K=16 means we only have k = 0-15, not k = 0-31!

    # Actually, for mfma_16x16x16_f16 with K=16:
    # Each lane needs 4 f16 = 2 u32
    # Lane distribution:
    # - Lanes 0-15: k_block = 0, t = 0-3, reads k = 0-3
    # - Lanes 16-31: k_block = 1, t = 0-3, reads k = 4-7
    # - Lanes 32-47: k_block = 2, t = 0-3, reads k = 8-11
    # - Lanes 48-63: k_block = 3, t = 0-3, reads k = 12-15

    # But wait, the kernel uses (2, 4, 1) view which expects 8 f16 per lane!
    # Let me check what the kernel expects...

    # Looking at the kernel:
    #   m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    # This views 4 u32 = 8 f16 as (2, 4, 1)
    # Then m_a[0] is the first 4 f16, m_a[1] is the next 4 f16

    # For K=32: m_a[0] covers k=0-15, m_a[1] covers k=16-31
    # For K=16: we only have 4 f16 = 2 u32 per lane, not enough!

    # So the current kernel REQUIRES K=32 (or we need to modify it).

    # Solution: For split-K with K_per_split = 16, we need a different kernel
    # that only makes ONE MFMA call with K=16.

    # OR: We can keep the K tiles at 32, and adjust the split-K partitioning.
    # Instead of splitting by channels, we split by K tiles.

    print("  Issue: kernel expects K=32, but K_per_split=16")
    print("  Need to modify the kernel for K=16 support")

print("\n" + "=" * 60)
print("Test 3: Modified approach - process K tiles, not channel splits")
print("=" * 60)

# Alternative: Process all K tiles and split the work differently
# For Conv2D with K = in_channels * kernel_area = 64 * 9 = 576
# With MFMA_K = 32, we have 576 / 32 = 18 K tiles

# Split-K approach: distribute K tiles across splits
# SPLIT_K_SLICES = 2 means first 9 tiles on split 0, next 9 on split 1

# But the original requirement says to split by channels:
# c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)
# This doesn't align with MFMA_K tiles!

# Let me check the original requirements...

print("Original split-K requirements:")
print("  c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)")
print("  K = in_channels * kernel_area")
print("  K tiles are groups of MFMA_K = 32 elements")
print()
print("For in_channels = 64, kernel_area = 9:")
print("  K = 64 * 9 = 576")
print("  K tiles = 576 / 32 = 18")
print("  c_per_split = 32")
print("  K per split = 32 * 9 = 288")
print("  K tiles per split = 288 / 32 = 9")
print()
print("This should work! Each split processes 9 complete K tiles.")

# The issue in my test is that K=32 is too small.
# Let me test with K=64 (2 K tiles) and SPLIT_K=2.

print("\n" + "=" * 60)
print("Test 4: K=64, SPLIT_K=2 (realistic for Conv2D)")
print("=" * 60)

K_test = 64
A_test = torch.randn((M, K_test), dtype=torch.float16, device=device)
B_test = torch.randn((K_test, N), dtype=torch.float16, device=device)

expected_test = torch.matmul(A_test.float(), B_test.float())

# Process with single kernel (2 K tiles = 2 x 2 MFMA calls = 4 MFMA total)
C_test_single = torch.zeros((M, N), dtype=torch.float32, device=device)

frag_size_full = 8  # 8 f16 per lane per K tile

for k_tile in range(K_test // 32):  # 2 K tiles
    k_start = k_tile * 32

    # Shuffle
    A_shuffled = torch.zeros((WARP_SIZE, frag_size_full), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size_full):
            k = k_block * frag_size_full + t
            A_shuffled[lane, t] = A_test[i, k_start + k]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size_full), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size_full):
            k = k_block * frag_size_full + t
            B_shuffled[lane, t] = B_test[k_start + k, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_full // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_full // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unshuffle and accumulate
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_test_single[4 * g + t, j] += C_shuffled[lane, t]

diff_single = torch.abs(C_test_single - expected_test)
print(f"Single kernel (K=64): max diff = {diff_single.max().item():.6f}")

# Now try split-K
print("\nSplit-K approach (K=64, SPLIT_K=2):")

SPLIT_K_test = 2
K_per_split_test = K_test // SPLIT_K_test  # = 32

C_test_split = torch.zeros((M, N), dtype=torch.float32, device=device)

for split_id in range(SPLIT_K_test):
    k_split_start = split_id * K_per_split_test
    k_split_end = k_split_start + K_per_split_test

    print(f"Split {split_id}: k = {k_split_start} to {k_split_end-1}")

    # Each split has exactly one K tile (K_per_split_test = 32)
    # Process with one kernel call

    A_shuffled = torch.zeros((WARP_SIZE, frag_size_full), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size_full):
            k = k_block * frag_size_full + t
            A_shuffled[lane, t] = A_test[i, k_split_start + k]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size_full), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size_full):
            k = k_block * frag_size_full + t
            B_shuffled[lane, t] = B_test[k_split_start + k, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_full // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size_full // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unshuffle and accumulate
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_test_split[4 * g + t, j] += C_shuffled[lane, t]

diff_split = torch.abs(C_test_split - expected_test)
print(f"Split-K (K=64): max diff = {diff_split.max().item():.6f}")
print(f"Result matches: {torch.allclose(C_test_split, expected_test, rtol=1e-2, atol=1e-3)}")
