import torch
import substrate
import substrate.language as S

# Test 2x2 MFMA tiles using same accumulator pattern

M = 64
N = 64
K = 8
WARP_SIZE = 64

@substrate.jit
def gemm_2x2_fixed_kernel(
    A: S.Tensor((64, 8), S.bf16),
    B: S.Tensor((8, 64), S.bf16),
    C: S.Tensor((64, 64), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Use a single accumulator that we reuse
    acc = S.full((16,), 0.0, S.f32)

    # === Tile (0,0): A[0:32, :] @ B[:, 0:32] ===
    a_frag = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = A[lane_col, k_idx]  # row = lane_col (0-31)
        b_frag[e] = B[k_idx, lane_col]  # col = lane_col (0-31)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Write tile (0,0)
    for acc_idx in S.range(16):
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col
        C[row, col] = acc[acc_idx]

    # Reset accumulator
    for i in S.range(16):
        acc[i] = S.convert(0.0, S.f32)

    # === Tile (0,1): A[0:32, :] @ B[:, 32:64] ===
    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = A[lane_col, k_idx]  # row = lane_col (0-31)
        b_frag[e] = B[k_idx, lane_col + 32]  # col = lane_col + 32 (32-63)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Write tile (0,1)
    for acc_idx in S.range(16):
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col + 32
        C[row, col] = acc[acc_idx]

    # Reset accumulator
    for i in S.range(16):
        acc[i] = S.convert(0.0, S.f32)

    # === Tile (1,0): A[32:64, :] @ B[:, 0:32] ===
    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = A[lane_col + 32, k_idx]  # row = lane_col + 32 (32-63)
        b_frag[e] = B[k_idx, lane_col]  # col = lane_col (0-31)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Write tile (1,0) - rows 32-63
    for acc_idx in S.range(16):
        row = 32 + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col
        C[row, col] = acc[acc_idx]

    # Reset accumulator
    for i in S.range(16):
        acc[i] = S.convert(0.0, S.f32)

    # === Tile (1,1): A[32:64, :] @ B[:, 32:64] ===
    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = A[lane_col + 32, k_idx]  # row = lane_col + 32 (32-63)
        b_frag[e] = B[k_idx, lane_col + 32]  # col = lane_col + 32 (32-63)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Write tile (1,1) - rows 32-63, cols 32-63
    for acc_idx in S.range(16):
        row = 32 + 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col + 32
        C[row, col] = acc[acc_idx]


if __name__ == "__main__":
    A = torch.randn((M, K), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((K, N), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    expected = A.float() @ B.float()

    gemm_2x2_fixed_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'2x2 tile fixed (same accumulator):')
    print(f'  Max diff: {max_diff}')
    print(f'  Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    # Check each tile
    print(f'\nTile (0,0): {torch.max(torch.abs(actual[:32, :32] - expected_cpu[:32, :32])).item():.6f}')
    print(f'Tile (0,1): {torch.max(torch.abs(actual[:32, 32:] - expected_cpu[:32, 32:])).item():.6f}')
    print(f'Tile (1,0): {torch.max(torch.abs(actual[32:, :32] - expected_cpu[32:, :32])).item():.6f}')
    print(f'Tile (1,1): {torch.max(torch.abs(actual[32:, 32:] - expected_cpu[32:, 32:])).item():.6f}')
