import torch
import substrate
import substrate.language as S

# Test ONLY tile (1,0) - rows 32-63, cols 0-31

M = 32
N = 32
K = 8
WARP_SIZE = 64

@substrate.jit
def gemm_tile10_kernel(
    A: S.Tensor((64, 8), S.f32),
    B: S.Tensor((64, 8), S.f32),
    C: S.Tensor((32, 32), S.f32),  # Output is 32x32
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    acc = S.full((16,), 0.0, S.f32)

    # Load A fragment for rows 32-63
    a_frag = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag[e] = S.convert(A[lane_col + 32, k_idx], S.bf16)

    # Load B fragment for columns 0-31
    b_frag = S.make_local((4,), S.bf16)
    for e in S.range(4):
        k_idx = lane_k_base + e
        b_frag[e] = S.convert(B[lane_col, k_idx], S.bf16)

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

    # Writeback
    for acc_idx in S.range(16):
        row = 16 * (lane // 32) + acc_idx
        col = lane_col
        C[row, col] = acc[acc_idx]


if __name__ == "__main__":
    A = torch.randn((64, 8), dtype=torch.float32, device='cuda')
    B = torch.randn((64, 8), dtype=torch.float32, device='cuda')
    C = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    # Expected: C = A[32:64, :] @ B[0:32, :].T
    expected = A[32:64, :] @ B[0:32, :].T

    gemm_tile10_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Single tile (1,0) test:')
    print(f'  Max diff: {max_diff}')
    print(f'  Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    # Compare with tile (0,0) using same B
    C00 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    @substrate.jit
    def gemm_tile00_kernel(
        A: S.Tensor((64, 8), S.f32),
        B: S.Tensor((64, 8), S.f32),
        C: S.Tensor((32, 32), S.f32),
    ):
        tid = S.thread_id(0)
        lane = tid % 64

        lane_col = lane % 32
        lane_k_base = (lane // 32) * 4

        acc = S.full((16,), 0.0, S.f32)

        # Load A fragment for rows 0-31
        a_frag = S.make_local((4,), S.bf16)
        for e in S.range(4):
            k_idx = lane_k_base + e
            a_frag[e] = S.convert(A[lane_col, k_idx], S.bf16)

        # Load B fragment for columns 0-31
        b_frag = S.make_local((4,), S.bf16)
        for e in S.range(4):
            k_idx = lane_k_base + e
            b_frag[e] = S.convert(B[lane_col, k_idx], S.bf16)

        acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag, b_frag, acc)

        for acc_idx in S.range(16):
            row = 16 * (lane // 32) + acc_idx
            col = lane_col
            C[row, col] = acc[acc_idx]

    gemm_tile00_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, B, C00)

    print(f'\nComparing tile (0,0) vs tile (1,0):')
    print(f'  Are they equal? {torch.allclose(C.cpu(), C00.cpu())}')
    print(f'  C00[0, :4]: {C00.cpu()[0, :4]}')
    print(f'  C10[0, :4]: {actual[0, :4]}')
