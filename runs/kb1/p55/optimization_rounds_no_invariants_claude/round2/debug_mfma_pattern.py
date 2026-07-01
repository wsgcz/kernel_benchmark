#!/usr/bin/env python3
"""Debug MFMA lane-to-matrix element mapping."""
import torch
import torch.nn as nn

# Smaller shape for testing
BATCH = 1
IN_CHANNELS = 16  # Small for testing
OUT_CHANNELS = 16
IN_H = 16
IN_W = 16
KERNEL_H = 3
KERNEL_W = 3
OUT_H = IN_H - KERNEL_H + 1  # 14
OUT_W = IN_W - KERNEL_W + 1  # 14
KERNEL_AREA = KERNEL_H * KERNEL_W  # 9
GEMM_K = IN_CHANNELS * KERNEL_AREA  # 144

# MFMA parameters
WARP_SIZE = 64
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32

print("=" * 60)
print("MFMA Lane-to-Matrix Element Mapping")
print("=" * 60)

# Show shuffle patterns
print("\nA matrix shuffle (from reference):")
print("For each lane: i = lane % 16, k_block = lane // 16")
print("A_shuffled[lane, t] = A[i, k_block*frag_size + t]")
print()
for lane in range(16):
    i = lane % MFMA_M
    k_block = lane // MFMA_M
    print(f"Lane {lane}: i={i}, k_block={k_block}, reads A row {i}")

print("\nB matrix shuffle (from reference):")
print("For each lane: k_block = lane // 16, j = lane % 16")
print("B_shuffled[lane, t] = B[k_block*frag_size + t, j]")
print()
for lane in range(16):
    k_block = lane // MFMA_N
    j = lane % MFMA_N
    print(f"Lane {lane}: k_block={k_block}, j={j}, reads B col {j}")

print("\nC matrix unshuffle (from reference):")
print("For each lane: g = lane // 16, j = lane % 16")
print("C[g*4+t, j] = C_shuffled[lane, t]")
print()
for lane in range(16):
    g = lane // MFMA_N
    j = lane % MFMA_N
    rows = [g * 4 + t for t in range(4)]
    print(f"Lane {lane}: g={g}, j={j}, writes C rows {rows}, col {j}")

print("\n" + "=" * 60)
print("Key insight: MFMA computes C[i, j] where i = lane % 16")
print("=" * 60)
print("""
The reference test shows:
- A shuffling: lane i reads A row i (for i = lane % 16)
- C unshuffling: lane i writes C rows [g*4+0, g*4+1, g*4+2, g*4+3] at col j

But MFMA computes: C = A @ B
If A is read at row i by lane (i % 16), then C should be at row i.

The unshuffle writes C rows [g*4+t] from lane (g*16 + j).
For lane j (where g=0): writes C rows [0,1,2,3]
For lane i: reads A row i

For MFMA to be correct:
- Lane that reads A row i should produce C row i
- Lane i (g=0, j=i) writes C rows [0,1,2,3] at col i

Wait, that means lane i writes MULTIPLE rows! Each lane produces 4 output values.
Let me re-examine the MFMA semantics...

The mfma_16x16x16_f16_f32 computes a 16x16x16 block matrix multiply.
Each lane holds a fragment of the result matrix.
The lane distribution for the result is:
- Lane j (j = lane % 16) holds column j
- Rows are distributed as: lane g (g = lane // 16) holds rows [g*4 : g*4+4]

So for lane i (where g = i // 16, j = i % 16):
- This lane writes to column j of rows [g*4, g*4+1, g*4+2, g*4+3]

For the input A:
- Lane i (i = lane % 16) reads row i
- But with lane = g*16 + j, we have i = j (lane % 16 = j)
- So lane (g*16 + j) reads A row j

This means:
- Lane (g*16 + j) reads A row j
- Lane (g*16 + j) writes C rows [g*4 : g*4+4] at col j

For MFMA to work: C[g*4+t, j] = sum_k A[j, k] * B[k, j]
But that's not how matrix multiply works!

Actually, let me re-read the reference more carefully...
""")

print("\n" + "=" * 60)
print("Re-reading reference test pattern")
print("=" * 60)

# From reference: A_shuffled[lane, t] = A[i, k]
# where i = lane % 16, k = (lane // 16) * frag_size + t
#
# For lane = 0: i=0, k=0,1,2,3 (reads A[0, 0:4])
# For lane = 1: i=1, k=0,1,2,3 (reads A[1, 0:4])
# ...
# For lane = 16: i=0, k=4,5,6,7 (reads A[0, 4:8])
# For lane = 17: i=1, k=4,5,6,7 (reads A[1, 4:8])

print("A shuffle pattern (detailed):")
print("Lane 0: reads A[0, 0:4]")
print("Lane 1: reads A[1, 0:4]")
print("...")
print("Lane 15: reads A[15, 0:4]")
print("Lane 16: reads A[0, 4:8]")
print("Lane 17: reads A[1, 4:8]")
print("...")
print("Lane 63: reads A[15, 28:32]")

print("\nB shuffle pattern (detailed):")
print("Lane 0: reads B[0:4, 0]")
print("Lane 1: reads B[0:4, 1]")
print("...")
print("Lane 15: reads B[0:4, 15]")
print("Lane 16: reads B[4:8, 0]")
print("...")
print("Lane 63: reads B[28:32, 15]")

print("\nC unshuffle pattern (detailed):")
print("Lane 0: writes C[0:4, 0]")
print("Lane 1: writes C[0:4, 1]")
print("...")
print("Lane 15: writes C[0:4, 15]")
print("Lane 16: writes C[4:8, 0]")
print("...")
print("Lane 63: writes C[12:16, 15]")

print("\n" + "=" * 60)
print("Checking consistency")
print("=" * 60)

# Check if the shuffling is consistent
# For C[0, 0]: written by lane 0, t=0 (row = 0, col = 0)
# For C[0, 0]: needs A[0, :] @ B[:, 0]
#
# A[0, :] is read by lanes 0, 16, 32, 48 (each reads 4 elements)
# B[:, 0] is read by lanes 0, 16, 32, 48 (each reads 4 elements)
#
# Lane 0 reads A[0, 0:4] and B[0:4, 0]
# Lane 16 reads A[0, 4:8] and B[4:8, 0]
# etc.
#
# After MFMA, each lane produces partial sum for C[0:4, col]
# But wait, that means C rows 0-3 are all produced by the same lane?
# No, let me re-check...

print("""
Let me verify with a concrete example:
- Lane 0: g=0, j=0
  - A shuffle: i=0, k_block=0 -> reads A[0, 0:4]
  - B shuffle: k_block=0, j=0 -> reads B[0:4, 0]
  - C unshuffle: g=0, j=0 -> writes C[0:4, 0]
  - So lane 0 computes partial sum for C[0:4, 0] using A[0, 0:4] and B[0:4, 0]?

But MFMA computes: C += A @ B
For C[0, 0], we need A[0, 0]*B[0, 0] + A[0, 1]*B[1, 0] + ...
Lane 0 has A[0, 0:4] and B[0:4, 0]
This gives: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]

But this is a partial sum! We need all K elements.
Lane 16 has A[0, 4:8] and B[4:8, 0]
This gives: A[0,4]*B[4,0] + A[0,5]*B[5,0] + A[0,6]*B[6,0] + A[0,7]*B[7,0]

So multiple lanes contribute to the same output element C[0, 0]?
No wait, that doesn't match the unshuffle pattern...

Let me re-think: MFMA_16x16x16 means:
- Computes a 16x16 output block from 16x16 A and 16x16 B
- Each MFMA call computes: C[0:16, 0:16] += A[0:16, 0:16] @ B[0:16, 0:16]

But the K dimension is 16 (one MFMA call), not 32 (two MFMA calls).
With two MFMA calls:
- First call: C += A[:, 0:16] @ B[0:16, :]
- Second call: C += A[:, 16:32] @ B[16:32, :]

For this to work, the A and B fragments need to be packed correctly.
""")

print("\n" + "=" * 60)
print("Understanding MFMA fragment layout")
print("=" * 60)

print("""
From AMD documentation and the reference test:
- mfma_16x16x16_f16_f32 computes: C += A @ B where A is 16x16, B is 16x16, C is 16x16
- Each lane holds 4 f32 values of C
- The C value distribution: lane g (g = lane // 16) holds rows [g*4 : g*4+4]
- Column j (j = lane % 16) is held by lane (g*16 + j) for all g

For lane (g, j):
- Holds C[g*4 : g*4+4, j]
- Needs to read A[g*4 : g*4+4, :] and B[:, j]

But wait, the shuffle pattern shows:
- Lane (g, j) reads A row j (from A shuffle: i = lane % 16 = j)
- Lane (g, j) reads B column j (from B shuffle: j = lane % 16)

This is INCONSISTENT!

Actually, I think I'm misreading the shuffle pattern. Let me look again...

From reference:
  i = lane % 16  # row index
  k_block = lane // 16
  for t in range(frag_size):
      k = k_block * frag_size + t
      A_shuffled[lane, t] = A[i, k]

So lane reads A[i, k_block*frag_size + t].
For lane = 0: i=0, k_block=0, reads A[0, 0:4]
For lane = 16: i=0, k_block=1, reads A[0, 4:8]
For lane = 32: i=0, k_block=2, reads A[0, 8:12]
For lane = 48: i=0, k_block=3, reads A[0, 12:16]

So lanes 0, 16, 32, 48 all read from A row 0!

For lane = 1: i=1, k_block=0, reads A[1, 0:4]
For lane = 17: i=1, k_block=1, reads A[1, 4:8]
...

And for B:
  k_block = lane // 16
  j = lane % 16
  for t in range(frag_size):
      k = k_block * frag_size + t
      B_shuffled[lane, t] = B[k, j]

Lane = 0: k_block=0, j=0, reads B[0:4, 0]
Lane = 16: k_block=1, j=0, reads B[4:8, 0]
Lane = 32: k_block=2, j=0, reads B[8:12, 0]
Lane = 48: k_block=3, j=0, reads B[12:16, 0]

So lanes 0, 16, 32, 48 all read from B column 0!

For C unshuffle:
  g = lane // 16
  j = lane % 16
  for t in range(4):
      row = g * 4 + t
      col = j
      C[row, col] = C_shuffled[lane, t]

Lane = 0: g=0, j=0, writes C[0:4, 0]
Lane = 16: g=1, j=0, writes C[4:8, 0]
Lane = 32: g=2, j=0, writes C[8:12, 0]
Lane = 48: g=3, j=0, writes C[12:16, 0]

So lanes 0, 16, 32, 48 all write to C column 0!

Now the pattern makes sense:
- Lane (g, j) holds C[g*4 : g*4+4, j]
- All lanes with same j read B column j
- All lanes with same g read A rows [g*4 : g*4+4]? NO!

Wait, the A shuffle reads A row i = lane % 16 = j.
So all lanes with same j read A row j!

This still doesn't match. Let me think about what the MFMA actually computes...

Actually, I think the MFMA operation works like this:
- The A fragment held by lane (g, j) is for A row (g*4 + t) where t varies
- But the shuffle is packing A row i into lane i

There must be some implicit transposition or different interpretation.

Let me just trust the reference test - it works! The key is that the shuffle/unshuffle
patterns are correct for the reference test.

My implementation follows the same pattern, so it should be correct...

Actually wait! I think I see the issue now!

In my implementation:
- _shuffle_a_fragment reads A row i (where i = lane % 16)
- But the A matrix is the GEMM A matrix: (batch * hw_out) x K

In the GEMM test, the A matrix has shape (16, 32) = (M, K).
M dimension is the row dimension (output spatial positions batched together).
K dimension is the column dimension (input channels * kernel area).

So when I shuffle A, I'm reading from the correct row (output spatial position).

But wait, my _shuffle_a_fragment uses:
  hw_idx = spatial_start + i
  where i = lane % MFMA_M

And for C unshuffle:
  hw_idx = spatial_start + row
  where row = g * 4 + t

So:
- Lane i reads from hw_idx = spatial_start + i (for i = lane % 16)
- Lane g*16+j writes to hw_idx = spatial_start + g*4+t

For lane 0: i=0, reads hw_idx = spatial_start + 0
            g=0, j=0, writes hw_idx = spatial_start + [0,1,2,3]

So lane 0 reads from row 0 but writes to rows [0,1,2,3]!

This is WRONG! The row indices don't match!

Actually, let me check the reference test again...
""")

# Let's run a simple test with the reference patterns
print("\n" + "=" * 60)
print("Running simple test with reference patterns")
print("=" * 60)

import sys
sys.path.insert(0, "/workspace/substrate/test/examples/gemm/amdgpu")
from test_gemm_mfma import gemm_mfma_16x16x16

if torch.cuda.is_available():
    device = "cuda"
    M, N, K = 16, 16, 32
    WARP_SIZE = 64

    # Create test matrices
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device=device)
    B = torch.randn((K, N), dtype=torch.float16, device=device)
    C = torch.zeros((M, N), dtype=torch.float16, device=device)

    expected = (A @ B).to(dtype=torch.float16, device="cpu")

    frag_size = K // 4  # 8 f16 elements per lane

    # Shuffle A (from reference)
    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            A_shuffled[lane, t] = A[i, k]

    # Shuffle B (from reference)
    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            B_shuffled[lane, t] = B[k, j]

    # Pack as u32
    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    C_shuffled = torch.zeros((WARP_SIZE, 4), dtype=torch.float32, device=device)

    # Run MFMA
    gemm_mfma_16x16x16[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    # Unshuffle C (from reference)
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C[4 * g + t, j] = C_shuffled[lane, t].to(torch.float16)

    actual = C.to("cpu")

    diff = torch.abs(actual - expected)
    print(f"Reference test result: max diff = {diff.max().item():.6f}")
    print(f"Result matches: {torch.allclose(actual, expected, rtol=1e-2, atol=1e-3)}")

    # Now let's verify the row mapping
    print("\nRow mapping verification:")
    print("Checking if the row that is read matches the row that is written...")

    # For each output element C[row, col], verify which lane produces it
    for row in range(16):
        for col in range(16):
            # Lane that writes C[row, col]
            g = row // 4
            t = row % 4
            j = col
            lane = g * 16 + j

            # A row that this lane reads
            i = lane % 16

            if row < 4 and col < 4:
                print(f"C[{row}, {col}]: written by lane {lane} (g={g}, j={j}, t={t}), reads A row {i}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("For C[row, col], the writing lane reads A row (lane % 16) = col")
    print("But C[row, col] should be computed from A[row, :] not A[col, :]")
    print("This means the MFMA is computing C.T (transpose) or there's a different interpretation!")
    print("=" * 60)

else:
    print("CUDA not available, skipping test")
