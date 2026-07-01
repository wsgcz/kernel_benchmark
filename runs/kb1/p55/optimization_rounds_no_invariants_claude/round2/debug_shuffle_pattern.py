#!/usr/bin/env python3
"""
Detailed analysis of MFMA shuffle patterns.

Key insight from reference test:
- Lane (g, j) where g = lane // 16, j = lane % 16
- Reads A row j, B column j
- Writes C rows [g*4 : g*4+4], column j

But for GEMM C[i, j] = sum_k A[i, k] * B[k, j]
C[0, 0] needs A[0, :] @ B[:, 0]
C[1, 0] needs A[1, :] @ B[:, 0]

So the lane pattern is:
- Lane (g, j) computes C[g*4+0:4, j] using A[0:16, :] and B[:, j]?

NO! Let me re-analyze...

The MFMA instruction computes a partial matrix multiply.
For a 16x16x16 MFMA:
- Each lane holds a fragment of the result
- The fragments are combined across lanes to produce the full 16x16 output

The key is that ALL lanes contribute to the result via the MFMA instruction.
Each lane holds a different part of the input matrices, but the result is computed
collectively.

Let me trace through exactly what happens for mfma_16x16x16_f16_f32:

Input fragments per lane:
- A fragment: 8 f16 values (4 u32)
- B fragment: 8 f16 values (4 u32)
- C fragment: 4 f32 values

The MFMA instruction computes:
- C_fragment += A_fragment @ B_fragment

But this is NOT a standard matrix multiply! The fragments are laid out in a specific
pattern defined by the hardware.

From AMD documentation, for mfma_16x16x16_f16:
- The 16x16 output is distributed across 64 lanes
- Each lane holds 4 elements: rows [g*4 : g*4+4], column j
- The A and B fragments are distributed such that the MFMA computes the correct
  partial sums across all lanes

So when I shuffle A and B according to the reference pattern, and run MFMA,
the hardware automatically combines the fragments from all 64 lanes to produce
the correct output.

The issue with my implementation is that I need to follow the EXACT shuffle pattern
from the reference. Let me compare:

Reference:
```python
for lane in range(WARP_SIZE):
    i = lane % 16  # row index
    k_block = lane // 16
    for t in range(frag_size):  # frag_size = 8
        k = k_block * frag_size + t  # k = k_block * 8 + t
        A_shuffled[lane, t] = A[i, k]
```

My implementation:
```python
for lane in range(WARP_SIZE):
    i = lane % MFMA_M  # row index (0-15)
    k_block = lane // MFMA_M  # which K block (0-3)

    for t in range(8):
        k = k_block * 8 + t
        k_global = k_start + k  # <-- DIFFERENT!
        ...
        A_shuffled[lane, t] = x_nchw[batch, c, ih, iw]
```

The difference is:
- Reference: A_shuffled[lane, t] = A[i, k] directly
- My code: A_shuffled[lane, t] = x_nchw[batch, c, ih, iw] where k_global = k_start + k

The issue is that I'm adding k_start to the local k index, which shifts the K dimension.
But the reference test doesn't use k_start because it processes the full K dimension in one go.

For tiling, I need to process K in tiles of MFMA_K=32. Each tile processes a different
K range. The MFMA instruction computes C += A_tile @ B_tile for each K tile.

So the correct approach is:
1. For each K tile starting at k_start:
   - Shuffle A[:, k_start : k_start+32] and B[k_start : k_start+32, :]
   - Run MFMA to accumulate C += A_tile @ B_tile

The shuffle pattern within each tile should follow the reference pattern, but with
the K indices shifted by k_start.

Let me verify my shuffle functions...

In my _shuffle_a_fragment:
```python
for t in range(8):
    k = k_block * 8 + t  # local K index within the tile (0-31)
    k_global = k_start + k  # global K index
    ...
```

This looks correct! For k_start=0, k_global = k, which matches the reference.
For k_start=32, k_global = 32 + k, which gives K indices 32-63.

But wait, the reference has frag_size = K // 4 = 32 // 4 = 8.
In the reference, K=32, so frag_size=8.

For my case, each K tile has K_tile = MFMA_K = 32.
So frag_size should be K_tile // 4 = 32 // 4 = 8.

This matches! Each lane has 8 f16 elements, which is 4 u32.

Let me check if the lane distribution is correct for the tile:
- Lanes 0-15: k_block = 0, reads k = 0-7
- Lanes 16-31: k_block = 1, reads k = 8-15
- Lanes 32-47: k_block = 2, reads k = 16-23
- Lanes 48-63: k_block = 3, reads k = 24-31

Wait, that's not right. Let me recalculate:
- Lane 0: i=0, k_block=0, reads A[0, 0:8]
- Lane 1: i=1, k_block=0, reads A[1, 0:8]
- ...
- Lane 15: i=15, k_block=0, reads A[15, 0:8]
- Lane 16: i=0, k_block=1, reads A[0, 8:16]
- ...
- Lane 31: i=15, k_block=1, reads A[15, 8:16]
- Lane 32: i=0, k_block=2, reads A[0, 16:24]
- ...
- Lane 47: i=15, k_block=2, reads A[15, 16:24]
- Lane 48: i=0, k_block=3, reads A[0, 24:32]
- ...
- Lane 63: i=15, k_block=3, reads A[15, 24:32]

So each lane reads 8 K elements. All 64 lanes together read:
- 16 rows x 32 columns = 512 f16 elements = 256 u32 = 64 lanes x 4 u32/lane ✓

The pattern is correct!

Now let me verify my unshuffle pattern:

My code:
```python
for lane in range(WARP_SIZE):
    g = lane // MFMA_N
    j = lane % MFMA_N

    for t in range(4):
        row = g * 4 + t
        col = j

        # Global indices in GEMM output
        hw_idx = spatial_start + row
        oc = oc_tile * MFMA_N + col

        if hw_idx < BATCH * OUT_H * OUT_W and oc < OUT_CHANNELS:
            y_gemm[hw_idx, oc] += C_shuffled[lane, t]
```

Reference:
```python
for lane in range(WARP_SIZE):
    g = lane // 16
    j = lane % 16
    for t in range(4):
        C[4 * g + t, j] = C_shuffled[lane, t].to(torch.float16)
```

My pattern matches the reference! The row is 4*g + t = g*4 + t, and col is j.

So the shuffle/unshuffle patterns should be correct. The issue must be elsewhere...

Let me think about what could be wrong:
1. The K linearization (k_global -> c, kh, kw)
2. The spatial position calculation (hw_idx -> batch, oh, ow)
3. The accumulation order

Let me trace through a concrete example...

Actually, I think I found the issue! Look at my unshuffle:

```python
y_gemm[hw_idx, oc] += C_shuffled[lane, t]
```

The += accumulates across splits, which is correct for split-K.

But I initialize C_shuffled to zeros for EACH output tile:
```python
C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)
```

And then I iterate over K tiles and accumulate:
```python
for k_start in range(k_start_global, k_end_global, MFMA_K):
    A_packed = _shuffle_a_fragment(...)
    B_packed = _shuffle_b_fragment(...)
    mfma_16x16x16_f16_kernel(A_packed, B_packed, C_shuffled)
```

The MFMA kernel reads C_shuffled, adds the result, and writes back. So after all
K tiles are processed, C_shuffled contains the accumulated result for this output
tile within this split.

Then I unshuffle and add to y_gemm:
```python
_unshuffle_c_to_gemm(C_shuffled, spatial_tile * MFMA_M, oc_tile, y_gemm, device)
```

This accumulates into y_gemm across splits. After all splits are processed, y_gemm
contains the full result.

This logic seems correct...

Let me check if there's an issue with the MFMA kernel itself. The kernel should
accumulate into C_shuffled correctly.

Looking at my kernel:
```python
c_lane = C[lane]  # Load current accumulator
...
c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
C[lane] = c_lane  # Store result
```

This loads C, executes two MFMA instructions, and stores back. This should accumulate
correctly across multiple kernel calls.

But wait, the kernel loads `c_lane = C[lane]` and then stores `C[lane] = c_lane`.
If the kernel is called multiple times, each call should accumulate.

Actually, I think the issue might be with how the kernel is launched. Let me check
the launch configuration:

```python
mfma_16x16x16_f16_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](
    A_packed, B_packed, C_shuffled
)
```

This launches with 1 block and WARP_SIZE threads. The kernel should read and write
C_shuffled correctly.

Hmm, let me add some debug output to see what's happening...

Actually, I think the issue might be with how I'm indexing the input data. Let me
trace through the indices more carefully.

In _shuffle_a_fragment:
```python
hw_idx = spatial_start + i
batch = hw_idx // (OUT_H * OUT_W)
hw_in_batch = hw_idx % (OUT_H * OUT_W)
oh = hw_in_batch // OUT_W
ow = hw_in_batch % OUT_W

...

A_shuffled[lane, t] = x_nchw[batch, c, ih, iw]
```

For spatial_start = 0 (first spatial tile):
- hw_idx = i (0-15)
- For i = 0: batch = 0, hw_in_batch = 0, oh = 0, ow = 0
- For i = 1: batch = 0, hw_in_batch = 1, oh = 0, ow = 1
- ...
- For i = 1021: batch = 0, hw_in_batch = 1021, oh = 0, ow = 1021
- For i = 1022: batch = 0, hw_in_batch = 1022, oh = 1, ow = 0

Wait, OUT_W = 1022, so:
- ow = hw_in_batch % OUT_W = hw_in_batch % 1022
- oh = hw_in_batch // OUT_W

For i = 0: oh = 0, ow = 0
For i = 1: oh = 0, ow = 1
...
For i = 1021: oh = 0, ow = 1021
For i = 1022: oh = 1, ow = 0

But i ranges from 0 to 15 for a single tile! So for spatial_tile = 0:
- i = 0 to 15 maps to (oh, ow) = (0, 0) to (0, 15)

This is correct! The first spatial tile covers the first 16 output pixels.

Let me check the input access:
```python
ih = oh + kh
iw = ow + kw
A_shuffled[lane, t] = x_nchw[batch, c, ih, iw]
```

This looks correct too.

Let me now check the weight access in _shuffle_b_fragment:
```python
oc = oc_tile * MFMA_N + col  # col = j = lane % 16
...
B_shuffled[lane, t] = w_oihw[oc, c, kh, kw]
```

Wait, this uses `col = j = lane % 16` and then `oc = oc_tile * 16 + col`.
For oc_tile = 0: oc = 0 to 15.
For oc_tile = 1: oc = 16 to 31.
...

And w_oihw[oc, c, kh, kw] is the weight for output channel oc, input channel c,
at kernel position (kh, kw).

This looks correct too!

Hmm, I'm not seeing an obvious bug. Let me run a minimal test case to isolate the issue.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

# Minimal test: Conv2D with tiny dimensions
print("=" * 60)
print("Minimal test with tiny dimensions")
print("=" * 60)

# Use the same dimensions as the reference GEMM test
M, N, K = 16, 16, 32
WARP_SIZE = 64

# Simulate a tiny Conv2D
# Input: (1, 32, 4, 4) -> output: (1, 16, 2, 2) with 3x3 kernel
# But for simplicity, use direct GEMM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create GEMM matrices
torch.manual_seed(0)
A = torch.randn((M, K), dtype=torch.float16, device=device)
B = torch.randn((K, N), dtype=torch.float16, device=device)

# Expected result
expected = (A @ B).to(device=device, dtype=torch.float32)

# Process with split-K
SPLIT_K_SLICES = 2
K_PER_SPLIT = K // SPLIT_K_SLICES

C = torch.zeros((M, N), dtype=torch.float32, device=device)

for split in range(SPLIT_K_SLICES):
    k_start = split * K_PER_SPLIT
    k_end = k_start + K_PER_SPLIT

    # Extract slices
    A_slice = A[:, k_start:k_end]
    B_slice = B[k_start:k_end, :]

    # For each K tile within the slice (assuming K_PER_SPLIT = 16, one tile)
    # If K_PER_SPLIT > MFMA_K, we'd need multiple tiles

    K_tile_size = min(K_PER_SPLIT, 16)  # MFMA_K for one 16x16x16 call

    if K_PER_SPLIT <= 16:
        # Single MFMA call
        frag_size = K_PER_SPLIT // 4

        A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            i = lane % 16
            k_block = lane // 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                if k < K_PER_SPLIT:
                    A_shuffled[lane, t] = A_slice[i, k]

        B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
        for lane in range(WARP_SIZE):
            k_block = lane // 16
            j = lane % 16
            for t in range(frag_size):
                k = k_block * frag_size + t
                if k < K_PER_SPLIT:
                    B_shuffled[lane, t] = B_slice[k, j]

        A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
        C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

        gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

        C_tile = torch.zeros((M, N), dtype=torch.float32, device=device)
        for lane in range(WARP_SIZE):
            g = lane // 16
            j = lane % 16
            for t in range(4):
                C_tile[4 * g + t, j] = C_shuffled[lane, t]

        C += C_tile

diff = torch.abs(C - expected)
print(f"Split-K GEMM result: max diff = {diff.max().item():.6f}")
print(f"Result matches: {torch.allclose(C, expected, rtol=1e-2, atol=1e-3)}")

# Compare with single-kernel approach
print("\n" + "=" * 60)
print("Single-kernel approach (K=16, no split)")
print("=" * 60)

# For K=16, we can use a single mfma_16x16x16_f16 call
# But wait, the reference test uses K=32 with two MFMA calls!

# Let me check the reference test again...
# The reference test has K=32 and makes two MFMA calls:
# c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
# c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)

# Each MFMA call processes K=16, so two calls process K=32 total.
# The A and B fragments are arranged as (2, 4, 1) f16, where:
# - m_a[0] and m_a[1] are the two halves of the A fragment
# - Each half has 4 f16 elements = K//4 = 32//4 = 8 f16 elements

# So for K=32:
# - Each lane has 8 f16 = 4 u32
# - Viewed as (2, 4, 1) f16, the two halves are m_a[0] and m_a[1]
# - Each half processes K=16

# For split-K with K=32 and SPLIT_K=2:
# - Each split has K_split = 16
# - This can be processed with a single MFMA call (or one half of the two-MFMA pattern)

# But my code uses MFMA_K = 32, which means each K tile is processed with two MFMA calls.
# For K_per_split = 16, I should only use one MFMA call!

# This might be the issue!

print("\n" + "=" * 60)
print("Testing with adjusted MFMA_K")
print("=" * 60)

# Let me test with K=16 per MFMA tile
K_MFMA = 16  # Each MFMA processes K=16

C2 = torch.zeros((M, N), dtype=torch.float32, device=device)

for split in range(SPLIT_K_SLICES):
    k_start = split * K_PER_SPLIT
    k_end = k_start + K_PER_SPLIT

    A_slice = A[:, k_start:k_end]
    B_slice = B[k_start:k_end, :]

    # Process with one MFMA call for K=16
    frag_size = K_PER_SPLIT // 4  # = 4

    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            if k < K_PER_SPLIT:
                A_shuffled[lane, t] = A_slice[i, k]

    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            if k < K_PER_SPLIT:
                B_shuffled[lane, t] = B_slice[k, j]

    # For K=16, we only need one MFMA call
    # But the kernel is written for K=32 (two MFMA calls)
    # We need to adjust or use a different kernel

    # Actually, let me check if frag_size = 4 works with the kernel...
    # The kernel views A[lane] as (2, 4, 1) f16
    # This means 2 x 4 = 8 f16 elements per lane
    # But with K=16, frag_size = 4, so we have only 4 f16 per lane!

    # The kernel expects 4 u32 = 8 f16 per lane.
    # With K=16, we have frag_size = 4 f16 = 2 u32 per lane.

    # This is a mismatch!

print("Mismatch detected: kernel expects K=32, but K_per_split=16")
print("Need to adjust the kernel or the split size!")
