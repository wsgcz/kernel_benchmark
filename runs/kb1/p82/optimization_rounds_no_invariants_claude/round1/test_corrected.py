#!/usr/bin/env python3
"""
Correct approach for depthwise conv with MFMA.

The key insight from analyzing the MFMA lane layout:
- Each lane l's B fragment contains B[k][l%16] for k = 4*(l//16) + t
- This is ONE COLUMN of the B matrix
- MFMA computes C = A * B using distributed fragments

For depthwise conv, we need:
C[i][j] = Σ_k A[i][k] * B[k][j]

If B is diagonal (B[k][j] = weight if k==j):
C[i][j] = A[i][j] * weight

But each lane has only ONE COLUMN of B (B[:][l%16]).
Lane l can only set B[:][l%16], not B[:][j] for arbitrary j.

Solution: Use MFMA where each lane's B column is the SAME weight.
B[k][j] = weight for ALL k, j.

Then: C[i][j] = Σ_k A[i][k] * weight = weight * Σ_k A[i][k]

This sums all A values, which is wrong!

Alternative: Set B[k][j] = weight for exactly one k value per column.
If we set B[k][j] = weight only when k = j:
- Lane 0 sets B[0][0] = weight (k=0, j=0)
- Lane 1 sets B[1][1] = weight (k=1, j=1)
- etc.

But lane 0's B fragment is B[:][0], not B[0][:].
Lane 0 has B[k][0] for k = 0,1,2,3.
To set B[0][0] = weight, lane 0 sets B_shmem[0, t=0] = weight.
This IS B[0][0]!

For lane 1 (j=1): B[k][1] for k = 0,1,2,3.
To set B[1][1] = weight, lane 1 needs k=1 in its k range.
Lane 1 has k_grp=0, k = 0,1,2,3. Yes, k=1 is available!
Lane 1 sets B_shmem[1, t=1] = weight, which is B[1][1]. ✓

So diagonal B CAN be constructed!

The issue is: when lane 0 computes C[0][1], it uses B[:][1], not B[:][0].
Lane 0's B fragment is B[:][0], so it can't see B[:][1] (from lane 1).

Wait, let me re-check the MFMA semantics...
MFMA computes C = A * B. Each lane contributes fragments of A and B.
The fragments are distributed across lanes, but the hardware combines them.

Actually, I think the issue is my understanding. Let me trace through the reference test again.

In the reference:
- A_shuffled[lane, t] = A[i, k] where i = lane%16, k = k_block*8 + t
- B_shuffled[lane, t] = B[k, j] where j = lane%16, k = k_block*8 + t

For K=32 with two MFMA calls:
- First MFMA: K=0..15, each lane has k = k_block*4 + t for k_block=0..3
- Second MFMA: K=16..31, each lane has k = k_block*4 + t + 16? No, different fragment indexing.

Looking at the MFMA call:
```python
m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
```

m_a[0] is the first 4 f16 values, m_a[1] is the second 4 f16 values.
Each 4 f16 values represent a "block" of K.

For the first MFMA call (m_a[0], m_b[0]):
- K indices are 0..15? Or just 0..3 per lane?

Actually, mfma_16x16x16 means K=16. Each MFMA call handles K=16.
With two calls, we handle K=32 total.

For the first call:
- Lane l holds A[l%16][4*(l//16)+t] for t=0..3
- Lane l holds B[4*(l//16)+t][l%16] for t=0..3

The K indices 0..15 are distributed:
- Lanes 0-15 have k=0..3
- Lanes 16-31 have k=4..7
- Lanes 32-47 have k=8..11
- Lanes 48-63 have k=12..15

So ALL 64 lanes contribute different k values for the single MFMA call.
The hardware combines these to compute the full K=16 sum.

This means: for C[i][j], the B[k][j] values come from lanes with the correct k.
- B[0][j] is in lane j (j in 0..15, k_grp=0)
- B[1][j] is in lane j (j in 0..15, k_grp=0)
- B[4][j] is in lane j+16 (j in 0..15, k_grp=1)
- etc.

So for B[k][j], the lane index is: lane = (k // 4) * 16 + j.

For diagonal B[k][j] = weight * δ[k,j]:
- B[0][0] = weight: lane = 0*16 + 0 = 0
- B[1][1] = weight: lane = 0*16 + 1 = 1
- B[4][4] = weight: lane = 1*16 + 4 = 20

Lane 20 has j = 20 % 16 = 4, k_grp = 20 // 16 = 1.
Lane 20's B fragment is B[k][4] for k = 4,5,6,7.
Setting B[4][4] = weight: lane 20 sets B_shmem[20, t=0] = weight.

Now, for computing C[0][4]:
- Lane 0 computes C[0][0..3], lane 16 computes C[0][4..7]
- Lane 16 has row_idx=0, k_grp=1, computes C[0][4], C[0][5], C[0][6], C[0][7]
- Lane 16's A fragment is A[0][k] for k=4,5,6,7
- Lane 16's B fragment is B[k][0] for k=4,5,6,7

For C[0][4] = Σ_k A[0][k] * B[k][4]:
- A[0][k] from lane 16: k=4,5,6,7
- A[0][k] from other lanes: k=0,1,2,3,8,9,... (via MFMA distribution)

Wait, the MFMA reads A fragments from ALL lanes, not just the current lane!
So when lane 16 executes MFMA:
- A fragments: A[i][k] from all lanes (i = lane%16, k = 4*(lane//16)+t)
- B fragments: B[k][j] from all lanes (k = 4*(lane//16)+t, j = lane%16)

For lane 16's MFMA:
- It reads A[0][k] for k=4..7 (from its own fragment)
- It reads B[k][0] for k=4..7 (from its own fragment)
- PLUS A and B from all other lanes!

The MFMA computes:
C[i][j] = Σ_k A[i][k] * B[k][j]

For i=0, j=4 (C[0][4]):
- A[0][k] from lane 0: k=0,1,2,3
- A[0][k] from lane 16: k=4,5,6,7
- A[0][k] from lane 32: k=8,9,10,11
- A[0][k] from lane 48: k=12,13,14,15

- B[k][4] from lane 4: k=0,1,2,3
- B[k][4] from lane 20: k=4,5,6,7
- B[k][4] from lane 36: k=8,9,10,11
- B[k][4] from lane 52: k=12,13,14,15

For diagonal B[k][j] = weight if k==j:
- B[4][4] from lane 20: k=4, j=4 → weight ✓
- Other B[k][4] = 0 (k≠4)

So C[0][4] = A[0][4] * B[4][4] = A[0][4] * weight.

The A[0][4] comes from lane 16 (row_idx=0, k_grp=1, k=4).
Lane 16 loads A[0][4] = X[0+k0][4+k1]... but wait, k=4 is used as column index in my current code!

Let me trace the A loading:
Lane 16: row_idx=0, k_grp=1
for t in 0..3:
  k = 4*1 + t = 4 + t
  i1 = k + k1  # column in input

So A[0][4] = X[0+k0][4+k1].

But for depthwise conv output C[0][4], we need X[0+k0][4+k1] * weight.
This matches! The input column is 4 + k1, which corresponds to output column 4.

So the mapping should be correct. The issue must be somewhere else...

Actually, wait. Let me re-check the B loading for diagonal.

Lane 20: row_idx=4, k_grp=1
for t in 0..3:
  k = 4*1 + t = 4,5,6,7
  j = row_idx = 4
  if k == j:
    B_shmem_f16[20, t] = weight  # t=0 for k=4
  else:
    B_shmem_f16[20, t] = 0

So B[4][4] = weight is set by lane 20. ✓

Lane 4: row_idx=4, k_grp=0
for t in 0..3:
  k = 4*0 + t = 0,1,2,3
  j = row_idx = 4
  if k == j:  # k=0..3, j=4, never true
    ...

Lane 4 sets B[0][4]=B[1][4]=B[2][4]=B[3][4]=0. ✓

So the diagonal B is constructed correctly!

For C[0][4]:
- Computed by lane 16
- Lane 16's acc[t] for t=0 gives C[0][4] (j = 4*1 + 0 = 4)
- C[0][4] = A[0][4] * B[4][4] = X[k0][4+k1] * weight

After 9 kernel positions: C[0][4] = Σ_{k0,k1} X[k0][4+k1] * W[k0][k1] ✓

This should be correct! Let me check what's actually happening in my code...

Oh wait, I see the issue! In my current code:
```python
if k == j:
    B_shmem_f16[lane, t] = w_f16
```

The condition is `k == j` where j = row_idx = lane % 16.
But k = 4 * k_grp + t = 4 * (lane // 16) + t.

For lane 0: k_grp=0, row_idx=0
- t=0: k=0, j=0 → k==j ✓
- t=1: k=1, j=0 → k≠j

For lane 1: k_grp=0, row_idx=1
- t=0: k=0, j=1 → k≠j
- t=1: k=1, j=1 → k==j ✓

This looks correct!

For lane 16: k_grp=1, row_idx=0
- t=0: k=4, j=0 → k≠j
- t=1: k=5, j=0 → k≠j

Lane 16 has j=0 (row_idx=0), but k=4..7. No diagonal elements.

Lane 20: k_grp=1, row_idx=4
- t=0: k=4, j=4 → k==j ✓
- t=1: k=5, j=4 → k≠j

Lane 20 sets B[4][4] = weight. ✓

So the B matrix IS correctly constructed!

The issue must be with how the A matrix is loaded or how the result is stored.

Let me check the A loading again:
```python
k = 4 * k_grp + t
i0 = row_idx + k0
i1 = k + k1  # Using k as column index
```

For lane 0: k_grp=0, row_idx=0
- t=0: k=0, i1=0+k1
- t=1: k=1, i1=1+k1

So A[0][0] = X[k0][0+k1], A[0][1] = X[k0][1+k1].

For depthwise conv output C[0][0]:
C[0][0] = Σ_k A[0][k] * B[k][0]

For diagonal B:
B[0][0] = weight (lane 0 sets this)
B[k][0] = 0 for k≠0

So C[0][0] = A[0][0] * B[0][0] = X[k0][0+k1] * weight.

Expected: C[0][0] = Σ_{k0,k1} X[k0][0+k1] * W[k0][k1].

But the current code gives: A[0][0] = X[k0][0+k1] for a specific kernel position.

After 9 kernel positions, C[0][0] = Σ_{positions} X[k0][0+k1] * W[k0][k1] ✓

This is correct! The first value (C[0][0]) matches in the test (1.1670).

The issue is with other values. Let me trace C[0][1]:
- Computed by lane 0, acc[1]
- C[0][1] = Σ_k A[0][k] * B[k][1]

A[0][k] for k=0..3 from lane 0:
A[0][0] = X[k0][0+k1], A[0][1] = X[k0][1+k1], etc.

B[k][1] for k=0..3:
B[0][1] from lane 1 (j=1, k_grp=0, k=0): B[0][1] = 0 (k≠j)
B[1][1] from lane 1 (j=1, k_grp=0, k=1): B[1][1] = weight ✓
B[2][1] from lane 1 (j=1, k_grp=0, k=2): B[2][1] = 0 (k≠j)
B[3][1] from lane 1 (j=1, k_grp=0, k=3): B[3][1] = 0 (k≠j)

So C[0][1] = A[0][1] * B[1][1] = X[k0][1+k1] * weight. ✓

After 9 kernel positions: C[0][1] = Σ_{k0,k1} X[k0][1+k1] * W[k0][k1] ✓

This should be correct! Why is the test failing?

Let me check the A loading more carefully. For C[0][1], we need A[0][1].
A[0][1] is loaded by lane 0 with k=1.
```python
i1 = k + k1 = 1 + k1
```
So A[0][1] = X[k0][1+k1]. ✓

Wait, but the output column for acc[1] is j = 4*k_grp + t = 4*0 + 1 = 1.
And the input column for A[0][1] is k + k1 = 1 + k1.

For depthwise conv output Y[0][1], we need input X[k0][1+k1].
The loaded input is X[k0][1+k1]. ✓

So everything should be correct!

Oh! I just realized the issue. When I compute C[i][j], I'm using B[k][j].
For C[0][1], I need B[:][1], but B[:][1] is stored by lane 1 (j=1).

When lane 0 executes MFMA, it reads B_shmem[0] (its own row), which is B[:][0].
Lane 0 does NOT read B_shmem[1] (which contains B[:][1]).

The MFMA hardware distributes B fragments, but each lane only provides ONE COLUMN of B.
The hardware must then broadcast these columns to all lanes.

So when lane 0 computes C, it should receive:
- B[:][0] from lane 0
- B[:][1] from lane 1
- B[:][2] from lane 2
- etc.

But in my shared memory implementation:
- Lane 0 writes B_shmem[0] = B[:][0]
- Lane 1 writes B_shmem[1] = B[:][1]
- Each lane reads only its own B_shmem[lane]

The issue: I'm using per-lane shared memory (B_shmem[lane]), but MFMA expects per-lane B fragments to represent DIFFERENT COLUMNS of the B matrix, not different elements of the same column.

When lane 0 calls MFMA with b_frag = view(B_shmem[0]), it's saying "my B column is B[:][0]".
The MFMA then uses this as the 0-th column of B for ALL rows of C.

So C[i][0] = Σ_k A[i][k] * B[k][0] (uses lane 0's B column)
C[i][1] = Σ_k A[i][k] * B[k][1] (uses lane 1's B column)
...

The MFMA hardware automatically routes B columns to the correct output columns!

So lane 0 computes C[:, 0..3] (multiple columns), but only provides B[:][0] (one column).
The B[:][1..3] columns come from lanes 1, 2, 3.

But my code has each lane reading from its own B_shmem[lane], not accessing other lanes' memory.
The MFMA instruction should handle this internally.

Actually, the MFMA instruction signature in the reference is:
```python
c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
```

The `m_b[0]` is the B fragment for that lane. The hardware uses all lanes' B fragments together.

So when lane 0 executes MFMA with m_b[0] = view(B_shmem[0]):
- It provides B[:][0] to the MFMA
- Lane 1 provides B[:][1]
- etc.
- The hardware combines these to compute all columns of C

For the result, lane 0 receives C[0][0..3] (4 values).
Lane 1 receives C[1][0..3].
...

Wait, that's the output distribution! Each lane gets 4 C values based on its row_idx.

So lane 0 receives:
- C[0][0], C[0][1], C[0][2], C[0][3] (row 0, columns based on k_grp)

Lane 16 receives:
- C[0][4], C[0][5], C[0][6], C[0][7] (row 0, columns based on k_grp=1)

This means the MFMA does work correctly across lanes!

So why is my test failing? Let me check the A loading again.

Actually, I think I see the issue now. In my A loading:
```python
i1 = k + k1
```

The column index is k, which is the K dimension index, not the output column j.

For lane 0 (k_grp=0):
- t=0: k=0, input column = 0+k1
- t=1: k=1, input column = 1+k1

Lane 0 computes C[0][0..3], where output columns are j=0,1,2,3.
But the loaded A values have input columns 0+k1, 1+k1, 2+k1, 3+k1.

For C[0][1] = A[0][1] * weight:
A[0][1] = X[k0][1+k1] ✓

This is correct! The input column 1+k1 corresponds to output column 1.

Hmm, the logic seems correct. Let me check if there's a bug in the actual code...

Actually, I wonder if the issue is with how I'm checking `if k == j`. Let me verify this is actually being executed correctly.

Let me add some debug output to see what's happening.
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def mfma_correct_kernel(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Correct MFMA kernel following reference pattern."""
    lane = S.thread_id(0)

    row_idx = lane % 16
    k_grp = lane // 16

    A_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    B_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    A_shmem_f16 = S.view(A_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    B_shmem_f16 = S.view(B_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    acc = S.full((4,), 0.0, S.f32)

    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load A: A[i][k] = X[i+k0][k+k1]
            # i = row_idx, k = 4*k_grp + t
            for t in S.range(4):
                k = 4 * k_grp + t
                i0 = row_idx + k0
                i1 = k + k1
                A_shmem_f16[lane, t] = X[0, 0, i0, i1]
                A_shmem_f16[lane, t + 4] = A_shmem_f16[lane, t]

            # Load B: diagonal B[k][j] = weight if k == j
            # Lane l holds B[k][row_idx] for k = 4*k_grp + t
            # j = row_idx for this lane
            for t in S.range(4):
                k = 4 * k_grp + t
                j = row_idx
                if k == j:
                    B_shmem_f16[lane, t] = w_f16
                else:
                    B_shmem_f16[lane, t] = S.convert(0.0, S.f16)
            for t in S.range(4, 8):
                B_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            S.syncthreads()

            a_frag = S.view(A_shmem[lane], S.Tensor((2, 4, 1), S.f16))
            b_frag = S.view(B_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag[0], acc)

    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running MFMA correct kernel...")
    try:
        mfma_correct_kernel[launch_config](x, w, y)
        print('Kernel completed!')

        ref_full = F.conv2d(x.float(), w.float()).half()
        ref = ref_full[:, :, :16, :16]

        diff = torch.abs(y - ref)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f'Max diff: {max_diff}')
        print(f'Mean diff: {mean_diff}')
        print(f'Y[0,0,0,:4]: {y[0,0,0,:4]}')
        print(f'Ref[0,0,0,:4]: {ref[0,0,0,:4]}')

        if torch.allclose(y, ref, rtol=1e-2, atol=0.1):
            print('SUCCESS!')
        else:
            print('MISMATCH')
    except Exception as e:
        import traceback
        traceback.print_exc()
