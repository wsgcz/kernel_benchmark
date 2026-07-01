import torch
import substrate
import substrate.language as S

# Test 2x2 MFMA using 2D tensor for accumulators (like conv2d_base.py)

M = 64
N = 64
K = 8
WARP_SIZE = 64
WAVE_REPEAT_M = 2
WAVE_REPEAT_N = 2
MFMA_ACC_SIZE = 16

@substrate.jit
def gemm_2x2_2d_acc_kernel(
    A: S.Tensor((64, 8), S.bf16),
    B: S.Tensor((8, 64), S.bf16),
    C: S.Tensor((64, 64), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Use 2D tensor for accumulators (like conv2d_base.py)
    acc = S.make_local((WAVE_REPEAT_M, WAVE_REPEAT_N, MFMA_ACC_SIZE), S.f32)

    # Initialize accumulators
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for i in S.range(MFMA_ACC_SIZE):
                acc[tm, tn, i] = S.convert(0.0, S.f32)

    # Compute all 4 tiles
    for tm in S.range(WAVE_REPEAT_M):
        a_row = lane_col + tm * 32
        a_frag = S.make_local((4,), S.bf16)

        for e in S.range(4):
            k_idx = lane_k_base + e
            a_frag[e] = A[a_row, k_idx]

        for tn in S.range(WAVE_REPEAT_N):
            b_col = lane_col + tn * 32
            b_frag = S.make_local((4,), S.bf16)

            for e in S.range(4):
                k_idx = lane_k_base + e
                b_frag[e] = B[k_idx, b_col]

            acc[tm, tn] = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc[tm, tn])

    # Writeback
    for tm in S.range(WAVE_REPEAT_M):
        for tn in S.range(WAVE_REPEAT_N):
            for acc_idx in S.range(MFMA_ACC_SIZE):
                row = tm * 32 + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
                col = tn * 32 + lane_col
                C[row, col] = acc[tm, tn, acc_idx]


if __name__ == "__main__":
    A = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((K, N), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    expected = A.float() @ B.float()

    gemm_2x2_2d_acc_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'2x2 tile with 2D accumulator:')
    print(f'  Max diff: {max_diff}')
    print(f'  Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    # Check each tile
    print(f'\nTile (0,0): {torch.max(torch.abs(actual[:32, :32] - expected_cpu[:32, :32])).item():.6f}')
    print(f'Tile (0,1): {torch.max(torch.abs(actual[:32, 32:] - expected_cpu[:32, 32:])).item():.6f}')
    print(f'Tile (1,0): {torch.max(torch.abs(actual[32:, :32] - expected_cpu[32:, :32])).item():.6f}')
    print(f'Tile (1,1): {torch.max(torch.abs(actual[32:, 32:] - expected_cpu[32:, 32:])).item():.6f}')
