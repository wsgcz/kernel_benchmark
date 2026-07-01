#!/usr/bin/env python3
"""Test if MFMA kernel modifies in-place or returns new tensor."""
import torch
import substrate
import substrate.language as S

WARP_SIZE = 64

@substrate.jit
def mfma_kernel(
    A: S.Tensor((64, 4), S.u32),
    B: S.Tensor((64, 4), S.u32),
    C: S.Tensor((64, 4), S.f32),
):
    lane = S.thread_id(0)
    c_lane = C[lane]
    m_a = S.view(A[lane], S.Tensor((2, 4, 1), S.f16))
    m_b = S.view(B[lane], S.Tensor((2, 4, 1), S.f16))
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[0], m_b[0], c_lane)
    c_lane = S.amdgpu.mfma_16x16x16_f16_f32(m_a[1], m_b[1], c_lane)
    C[lane] = c_lane


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test inputs
    A = torch.randn((64, 8), dtype=torch.float16, device=device)
    B = torch.randn((64, 8), dtype=torch.float16, device=device)

    # Pack as u32
    A_packed = A.view(torch.int32).view(64, 4)
    B_packed = B.view(torch.int32).view(64, 4)

    # Test 1: Call without return capture
    print("Test 1: Call without return capture")
    C1 = torch.zeros((64, 4), dtype=torch.float32, device=device)
    C1_id_before = id(C1)
    mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C1)
    print(f"C1 id before: {C1_id_before}, after: {id(C1)}")
    print(f"C1 modified in-place: {id(C1) == C1_id_before}")
    print(f"C1 sum: {C1.sum().item():.4f}")

    # Test 2: Call with return capture
    print("\nTest 2: Call with return capture")
    C2 = torch.zeros((64, 4), dtype=torch.float32, device=device)
    C2_id_before = id(C2)
    C2_result = mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C2)
    print(f"C2 id before: {C2_id_before}, result id: {id(C2_result)}")
    print(f"C2 is C2_result: {C2 is C2_result}")
    print(f"C2 sum: {C2.sum().item():.4f}")
    print(f"C2_result sum: {C2_result.sum().item():.4f}")

    # Test 3: Two MFMA calls accumulating
    print("\nTest 3: Two MFMA calls accumulating")
    C3 = torch.zeros((64, 4), dtype=torch.float32, device=device)
    mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C3)
    mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C3)
    print(f"C3 sum after two calls: {C3.sum().item():.4f}")
    print(f"Expected: ~2x of single call")

    # Test 4: Two MFMA calls with return capture
    print("\nTest 4: Two MFMA calls with return capture")
    C4 = torch.zeros((64, 4), dtype=torch.float32, device=device)
    C4 = mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C4)
    C4 = mfma_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A_packed, B_packed, C4)
    print(f"C4 sum after two calls: {C4.sum().item():.4f}")
    print(f"Expected: ~2x of single call")
