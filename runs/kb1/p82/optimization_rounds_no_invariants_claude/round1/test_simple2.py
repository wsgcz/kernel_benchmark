#!/usr/bin/env python3
"""Test with simpler weight loading."""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def simple_mfma_kernel_v2(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Simple 1-tile MFMA kernel with different weight loading."""
    lane = S.thread_id(0)

    o0_base = 0
    o1_base = 0

    # MFMA lane layout: lane l holds A[l%16][4*(l//16)+t]
    row_idx = lane % 16
    col_grp = lane // 16

    # Shared memory for staging
    input_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    weight_shmem = S.make_shared((WARP_SIZE, 4), S.u32)
    input_shmem_f16 = S.view(input_shmem, S.Tensor((WARP_SIZE, 8), S.f16))
    weight_shmem_f16 = S.view(weight_shmem, S.Tensor((WARP_SIZE, 8), S.f16))

    # Accumulator
    acc = S.full((4,), 0.0, S.f32)

    # Iterate over kernel positions
    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]

            # Load input: A[i][j] = X[i+k0][j+k1]
            # Lane l holds A[row_idx][4*col_grp + t]
            for t in S.range(4):
                col_out = 4 * col_grp + t  # j
                i0 = o0_base + row_idx + k0
                i1 = o1_base + col_out + k1

                input_shmem_f16[lane, t] = X[0, 0, i0, i1]
                input_shmem_f16[lane, t + 4] = input_shmem_f16[lane, t]

            # Weight loading: B[k][j] = weight when k == j (diagonal)
            # Lane l holds B[k][j] where k = 4*col_grp + t, j = row_idx
            # For diagonal: set B[j][j] = weight
            for t in S.range(4):
                k_val = 4 * col_grp + t
                j = row_idx
                # Only set diagonal elements: B[j][j]
                # Lane l can write B[k][j]. For diagonal, need k == j.
                if k_val == j:
                    weight_shmem_f16[lane, t] = w_f16
                else:
                    weight_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            for t in S.range(4, 8):
                weight_shmem_f16[lane, t] = S.convert(0.0, S.f16)

            S.syncthreads()

            # Perform MFMA
            a_frag = S.view(input_shmem[lane], S.Tensor((2, 4, 1), S.f16))
            b_frag = S.view(weight_shmem[lane], S.Tensor((2, 4, 1), S.f16))

            acc = S.amdgpu.mfma_16x16x16_f16_f32(a_frag[0], b_frag[0], acc)

    # Store output
    # Lane l holds C[row_idx][4*col_grp + t]
    for t in S.range(4):
        o0_out = o0_base + row_idx
        o1_out = o1_base + (4 * col_grp + t)
        Y[0, 0, o0_out, o1_out] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running kernel...")
    try:
        simple_mfma_kernel_v2[launch_config](x, w, y)
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
            # Print more details
            print(f'\nY[0,0,:4,:4]:\n{y[0,0,:4,:4]}')
            print(f'\nRef[0,0,:4,:4]:\n{ref[0,0,:4,:4]}')
    except Exception as e:
        import traceback
        traceback.print_exc()
