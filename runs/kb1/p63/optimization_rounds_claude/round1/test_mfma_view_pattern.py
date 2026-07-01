import torch
import substrate
import substrate.language as S

# Test two MFMA operations using the same pattern as amdgpu_conv2d_base.py

M = 32
N = 32
K = 8
WARP_SIZE = 64
MFMA_ACC_SIZE = 16

@substrate.jit
def two_mfma_kernel(
    A0: S.Tensor((32, 8), S.bf16),
    A1: S.Tensor((32, 8), S.bf16),
    B: S.Tensor((8, 32), S.bf16),
    C0: S.Tensor((32, 32), S.f32),
    C1: S.Tensor((32, 32), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Initialize accumulators using S.make_local
    acc0 = S.make_local((MFMA_ACC_SIZE,), S.f32)
    acc1 = S.make_local((MFMA_ACC_SIZE,), S.f32)

    for i in S.range(MFMA_ACC_SIZE):
        acc0[i] = S.convert(0.0, S.f32)
        acc1[i] = S.convert(0.0, S.f32)

    # Load fragments
    a_frag0 = S.make_local((4,), S.bf16)
    a_frag1 = S.make_local((4,), S.bf16)
    b_frag = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k_idx = lane_k_base + e
        a_frag0[e] = A0[lane_col, k_idx]
        a_frag1[e] = A1[lane_col, k_idx]
        b_frag[e] = B[k_idx, lane_col]

    # Use S.view to reshape for MFMA (like conv2d_base does)
    a_frag0_view = S.view(a_frag0, S.Tensor((1, 4, 1), S.bf16))
    a_frag1_view = S.view(a_frag1, S.Tensor((1, 4, 1), S.bf16))
    b_frag_view = S.view(b_frag, S.Tensor((1, 4, 1), S.bf16))

    acc0 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag0_view[0], b_frag_view[0], acc0)
    acc1 = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag1_view[0], b_frag_view[0], acc1)

    # Writeback using the working formula from test_gemm3.py
    for acc_idx in S.range(16):
        row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
        col = lane_col
        C0[row, col] = acc0[acc_idx]
        C1[row, col] = acc1[acc_idx]


if __name__ == "__main__":
    A0 = torch.randn((32, 8), dtype=torch.bfloat16, device='cuda')
    A1 = torch.randn((32, 8), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((8, 32), dtype=torch.bfloat16, device='cuda')
    C0 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')
    C1 = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    expected0 = A0.float() @ B.float()
    expected1 = A1.float() @ B.float()

    two_mfma_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A0, A1, B, C0, C1)

    actual0 = C0.cpu()
    actual1 = C1.cpu()
    expected0_cpu = expected0.cpu()
    expected1_cpu = expected1.cpu()

    diff0 = torch.max(torch.abs(actual0 - expected0_cpu)).item()
    diff1 = torch.max(torch.abs(actual1 - expected1_cpu)).item()

    print(f'Two MFMA test with view pattern:')
    print(f'  C0 (A0 @ B): max_diff={diff0:.6f}, pass={torch.allclose(actual0, expected0_cpu, rtol=1e-2, atol=0.1)}')
    print(f'  C1 (A1 @ B): max_diff={diff1:.6f}, pass={torch.allclose(actual1, expected1_cpu, rtol=1e-2, atol=0.1)}')

    print(f'\n  C0 == C1? {torch.allclose(actual0, actual1)}')
    print(f'  C0[0, :4] = {actual0[0, :4].tolist()}')
    print(f'  C1[0, :4] = {actual1[0, :4].tolist()}')
