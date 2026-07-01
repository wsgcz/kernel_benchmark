#!/usr/bin/env python3
"""
Direct element-wise multiply-accumulate for depthwise conv.

Since we can't efficiently construct a diagonal B matrix in shared memory,
we'll compute the element-wise operation directly in the accumulator.

For each kernel position:
1. Load input patch into A
2. Multiply each element by weight (in registers)
3. Add to accumulator

This avoids the MFMA K-reduction issue.
"""
import torch
import torch.nn.functional as F
import substrate
import substrate.language as S

WARP_SIZE = 64

def launch_config():
    return ((1, 1, 1), (WARP_SIZE, 1, 1))

@substrate.jit
def direct_depthwise_kernel(
    X: S.Tensor((1, 1, 20, 20), S.f16),
    W: S.Tensor((1, 1, 3, 3), S.f16),
    Y: S.Tensor((1, 1, 16, 16), S.f16),
):
    """Direct element-wise computation for depthwise conv."""
    lane = S.thread_id(0)

    row_idx = lane % 16
    k_grp = lane // 16

    # Accumulator for output (4 elements per lane)
    acc = S.full((4,), 0.0, S.f32)

    # For each kernel position
    for k0 in S.range(3):
        for k1 in S.range(3):
            w_f16 = W[0, 0, k0, k1]
            w_f32 = S.convert(w_f16, S.f32)

            # Load input and multiply by weight directly
            # Each lane loads 4 input values
            for t in S.range(4):
                j = 4 * k_grp + t
                i0 = row_idx + k0
                i1 = j + k1

                # Load input as f16, convert to f32, multiply by weight
                x_f16 = X[0, 0, i0, i1]
                x_f32 = S.convert(x_f16, S.f32)
                acc[t] = acc[t] + x_f32 * w_f32

    # Store output
    for t in S.range(4):
        j = 4 * k_grp + t
        Y[0, 0, row_idx, j] = S.convert(acc[t], S.f16)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
    w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')
    y = torch.zeros((1, 1, 16, 16), dtype=torch.float16, device='cuda')

    print("Running direct depthwise kernel (no MFMA)...")
    try:
        direct_depthwise_kernel[launch_config](x, w, y)
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
