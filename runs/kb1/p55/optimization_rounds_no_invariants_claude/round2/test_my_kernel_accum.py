#!/usr/bin/env python3
"""Test if my MFMA kernel accumulates from input C."""
import torch
import substrate
import substrate.language as S

WARP_SIZE = 64

@substrate.jit
def my_mfma_kernel(
    A: S.Tensor((64, 4), S.u32),
    B: S.Tensor((64, 4), S.u32),
    C: S.Tensor((64, 4), S.f32),
):
    lane = S.thread_id(0)
    c_lane = C[lane]  # LOAD from C - should accumulate
    m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
    C[lane] = c_lane


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frag_size = 8

    # Test accumulation
    A = torch.randn((16, 32), dtype=torch.float16, device=device)
    B = torch.randn((32, 16), dtype=torch.float16, device=device)
    expected = torch.matmul(A, B)

    # Pre-init accumulator with 10s
    C_shuffled = torch.ones((WARP_SIZE, 4), dtype=torch.float32, device=device) * 10.0

    # Shuffle and pack A
    A_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        i = lane % 16
        k_block = lane // 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            A_shuffled[lane, t] = A[i, k]

    # Shuffle and pack B
    B_shuffled = torch.zeros((WARP_SIZE, frag_size), dtype=torch.float16, device=device)
    for lane in range(WARP_SIZE):
        k_block = lane // 16
        j = lane % 16
        for t in range(frag_size):
            k = k_block * frag_size + t
            B_shuffled[lane, t] = B[k, j]

    A_packed = A_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)
    B_packed = B_shuffled.view(torch.int32).view(WARP_SIZE, frag_size // 2)

    print("Testing my kernel with pre-init accumulator...")
    print(f"C_shuffled[0,0] before: {C_shuffled[0,0].item():.4f}")

    # Call kernel with pre-init accumulator
    my_mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C_shuffled)

    print(f"C_shuffled[0,0] after: {C_shuffled[0,0].item():.4f}")

    # Unshuffle
    C_result = torch.zeros((16, 16), dtype=torch.float32, device=device)
    for lane in range(WARP_SIZE):
        g = lane // 16
        j = lane % 16
        for t in range(4):
            C_result[4 * g + t, j] = C_shuffled[lane, t]

    print(f'C_result[0,0] = {C_result[0,0].item():.4f}')
    print(f'expected[0,0] = {expected[0,0].item():.4f}')
    print(f'expected + 10 = {expected[0,0].item() + 10:.4f}')
    print()
    print('If MFMA accumulates: C_result should be expected + 10')
    print('If MFMA overwrites: C_result should be expected')
    print()

    diff_vs_expected = abs(C_result[0,0].item() - expected[0,0].item())
    diff_vs_expected_plus_10 = abs(C_result[0,0].item() - (expected[0,0].item() + 10))

    print(f'Diff vs expected: {diff_vs_expected:.4f}')
    print(f'Diff vs expected+10: {diff_vs_expected_plus_10:.4f}')

    if diff_vs_expected_plus_10 < diff_vs_expected:
        print("RESULT: MFMA ACCUMULATES correctly!")
    else:
        print("RESULT: MFMA OVERWRITES (does not accumulate)")
