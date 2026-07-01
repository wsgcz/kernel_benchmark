import torch
import substrate
import substrate.language as S

# Simplified Conv2D: 1x16x4x4 input, 4 output channels, 3x3 kernel
# Output: 1x4x2x2

N = 1
IN_CHANNELS = 16
IN_H = 4
IN_W = 4
OUT_CHANNELS = 4
KERNEL_H = 3
KERNEL_W = 3
OUT_H = 2
OUT_W = 2
KERNEL_AREA = KERNEL_H * KERNEL_W

# MFMA tiling
BLOCK_M = 4  # M_FLAT = 1 * 2 * 2 = 4
BLOCK_N = 4
WARP_SIZE = 64
NUM_WARPS = 1
THREADS_PER_BLOCK = WARP_SIZE * NUM_WARPS

HW_OUT = OUT_H * OUT_W
K_FLAT = IN_CHANNELS * KERNEL_AREA
M_FLAT = N * HW_OUT

@substrate.jit
def simple_conv_kernel(
    X: S.Tensor((1, 16, 4, 4), S.f32),
    W: S.Tensor((4, 16, 3, 3), S.f32),
    Y: S.Tensor((1, 4, 2, 2), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    group_m_base = S.block_id(0) * BLOCK_M
    group_n_base = S.block_id(1) * BLOCK_N

    acc = S.full((16,), 0.0, S.f32)

    # K tiles: K_FLAT = 144, MFMA_TILE_K = 8, so 18 tiles
    for k_tile in S.range(18):
        a_frag = S.make_local((4,), S.bf16)
        b_frag = S.make_local((4,), S.bf16)

        for e in S.range(4):
            a_frag[e] = S.convert(0.0, S.bf16)
            b_frag[e] = S.convert(0.0, S.bf16)

        m0 = group_m_base + lane_col
        n0 = group_n_base + lane_col

        for e in S.range(4):
            k_idx = k_tile * 8 + lane_k_base + e

            ic = k_idx // KERNEL_AREA
            spatial = k_idx % KERNEL_AREA
            kh = spatial // KERNEL_W
            kw = spatial % KERNEL_W

            if m0 < M_FLAT:
                batch0 = m0 // HW_OUT
                hw0 = m0 % HW_OUT
                oh0 = hw0 // OUT_W
                ow0 = hw0 % OUT_W
                ih0 = oh0 + kh
                iw0 = ow0 + kw
                a_frag[e] = S.convert(X[batch0, ic, ih0, iw0], S.bf16)

            if n0 < OUT_CHANNELS:
                b_frag[e] = S.convert(W[n0, ic, kh, kw], S.bf16)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Writeback: correct formula for MFMA 32x32x8 output layout
    for acc_idx in S.range(16):
        row = group_m_base + 16 * (lane // 32) + acc_idx
        col = group_n_base + lane_col

        if row < M_FLAT and col < OUT_CHANNELS:
            batch = row // HW_OUT
            hw = row % HW_OUT
            oh = hw // OUT_W
            ow = hw % OUT_W
            Y[batch, col, oh, ow] = acc[acc_idx]


if __name__ == "__main__":
    import torch.nn.functional as F

    x = torch.randn((1, 16, 4, 4), dtype=torch.float32, device='cuda')
    w = torch.randn((4, 16, 3, 3), dtype=torch.float32, device='cuda')
    y = torch.zeros((1, 4, 2, 2), dtype=torch.float32, device='cuda')

    expected = F.conv2d(x, w, stride=1, padding=0)

    simple_conv_kernel[lambda: ((1, 1, 1), (64, 1, 1))](x, w, y)

    actual = y.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    print(f'actual: {actual}')
    print(f'expected: {expected_cpu}')
