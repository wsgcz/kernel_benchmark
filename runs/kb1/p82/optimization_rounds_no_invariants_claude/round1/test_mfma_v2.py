#!/usr/bin/env python3
"""
Correct MFMA approach for depthwise conv.

Key insight from MFMA lane layout:
- Lane l holds B[k][j] where k = 4*(l//16) + t and j = l % 16
- Lane l computes C[i][j] where i = l % 16 and j = 4*(l//16) + t

For computing C[i][j] for multiple j values, lane l needs B[:][j] for those j.
But the MFMA fragment only contains B[k][j] for fixed j = l % 16.

This means MFMA naturally groups output columns:
- Lanes 0-15: j=0 for B, compute C[:][0..3]
- Lanes 16-31: j=1 for B, compute C[:][4..7]
- etc.

Wait, that doesn't match! Let me re-examine.

For MFMA_16x16x16 with 64 lanes:
- Each lane has 4 accumulator elements
- Lane l computes C[i][j] where i = l % 16, j = 4*(l//16) + t

So:
- Lanes 0-15 compute C[0..15][0..3] (first 4 columns)
- Lanes 16-31 compute C[0..15][4..7] (next 4 columns)
- etc.

For B matrix:
- Lane l holds B[k][j] where k = 4*(l//16) + t, j = l % 16
- Lanes 0-15 hold B[:][0] (first column of B)
- Lanes 16-31 hold B[:][1] (second column of B)
- etc.

But lanes 0-15 compute columns 0-3 of C, which need B[:][0], B[:][1], B[:][2], B[:][3]!
Each lane in 0-15 only has B[:][j=l%16], which is B[:][0..15] distributed across lanes.

The MFMA operation reads B fragments from all lanes and uses them for the matrix multiply.
C[i][j] = Σ_k A[i][k] * B[k][j]

For lanes 0-15 computing C[:][0..3]:
- C[:, 0] uses B[:][0] (from lanes 0-15, j=0)
- C[:, 1] uses B[:][1] (from lanes 16-31, j=1)
- C[:, 2] uses B[:][2] (from lanes 32-47, j=2)
- C[:, 3] uses B[:][3] (from lanes 48-63, j=3)

So lanes 0-15 compute C[:, 0..3], but they use B[:][0] from themselves and B[:][1..3] from other lanes!

This means the MFMA hardware automatically broadcasts B columns across lanes.
I don't need to worry about which lane has which B column - the hardware handles it!

Let me verify: for C[0][1], computed by lane 0 (t=1):
- Uses B[:][1] which is stored by lanes 16-31
- The MFMA instruction reads B[:][1] from those lanes

So I should:
1. Load A[i][k] correctly (each lane loads its own A values)
2. Set B[k][j] correctly (each lane sets its own B column)

For diagonal B: B[k][j] = weight if k == j, else 0
Lane l (with j = l % 16) should set:
- B[k][j] = weight if k == j
- But lane l can only write k in [4*(l//16), 4*(l//16)+3]

For lane 0 (j=0): can write B[0][0], B[1][0], B[2][0], B[3][0]
  Diagonal: B[0][0] = weight ✓
For lane 16 (j=1): can write B[4][1], B[5][1], B[6][1], B[7][1]
  Diagonal: B[1][1] = weight... but can lane 16 write k=1? No! Lane 16 has k_grp=1, k in [4..7].

So diagonal B[1][1] needs to be written by lane with j=1 and k=1.
That's lane 1 (j=1, k_grp=0, k in [0..3])!

Lane 1 can write B[1][1] = weight ✓

And lane 1 computes C[1][0..3], not C[0][1].
But C[0][1] is computed by lane 0, using B[:][1] from lanes 16-31.
Lanes 16-31 write B[k][1] for k in [4..7], not k=1.
So B[1][1] is written by lane 1, but lane 0 needs it!

The MFMA hardware should broadcast B[:][1] from lanes with j=1 (lanes 16-31) to all lanes.
But lane 1 (j=1) is in lanes 0-15, not lanes 16-31!

Wait, I'm confusing j values. Let me clarify:
- For B matrix: lane l writes B[k][j] where j = l % 16
- Lane 0: j=0, Lane 1: j=1, ..., Lane 15: j=15, Lane 16: j=0, ...
- So lanes 0, 16, 32, 48 all have j=0 for B!
- Lanes 1, 17, 33, 49 all have j=1 for B!

So B[:][0] is written by lanes 0, 16, 32, 48 (different k values).
B[:][1] is written by lanes 1, 17, 33, 49.
...

For diagonal B[j][j] = weight:
- B[0][0] needs lane with j=0 and k=0: Lane 0 ✓
- B[1][1] needs lane with j=1 and k=1: Lane 1 ✓
- B[4][4] needs lane with j=4 and k=4: Lane 20 (j=4, k_grp=1, k in [4..7]) ✓

So diagonal B CAN be constructed with the lane layout!

For C[0][1]:
- Computed by lane 0 (t=1)
- Uses B[:][1] from lanes with j=1: lanes 1, 17, 33, 49
- B[1][1] is written by lane 1 ✓
- Other B[k][1] for k!=1 should be 0

This should work! Let me re-examine my code...
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def mfma_depthwise_v2(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """MFMA depthwise conv with correct B matrix construction."""
    lane = S.thread_id(0)

    row_idx = lane % 16
    k_grp = lane // 16

    # Shared memory for A fragment
    input_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    input_shmem_f16 = S.view(input_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    # B fragment in registers (not shared memory)
    # Each lane holds B[k][j] for k = 4*k_grp + t, j = row_idx

    acc = S.full((4,), 0.0, S.f32)

    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load A[i][j] = X[i+k0][j+k1]
            for t in S.range(4):
                j = 4 * k_grp + t
                i0 = row_idx + k0
                i1 = j + k1
                input_shmem_f16[lane, t] = X[0, 0, i0, i1]
                input_shmem_f16[lane, t + 4] = input_shmem_f16[lane, t]

            # Create B fragment in registers
            # Lane l holds B[k][j] where k = 4*k_grp + t, j = row_idx
            # For diagonal: B[k][j] = weight if k == j, else 0
            b_frag = S.full((4,), 0.0, S.f16)
            for t in S.range(4):
                k = 4 * k_grp + t
                j = row_idx
                if k == j:
                    b_frag[t] = w_f16

            # Wait for A to be loaded
            S.syncthreads()

            # Create MFMA fragments
            a_frag = S.view(input_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            # Need to convert b_frag to the right format for MFMA
            # MFMA expects (2, 4, 1) f16 = 8 f16 values
            # We'll create a u32 array and view as f16
            b_storage = S.full((4,), 0, S.u32)
            b_storage_f16 = S.view(b_storage, S.Tensor((8,), S.f16))
            for t in S.range(4):
                b_storage_f16[t] = b_frag[t]
            for t in S.range(4, 8):
                b_storage_f16[t] = S.convert(0.0, S.f16)

            b_frag_view = S.view(b_storage, S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag_view[0], acc)

    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running MFMA depthwise v2...")
    try:
        mfma_depthwise_v2[launch_config](x, w, y)
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
