import torch
import substrate
import substrate.language as S

# Simple 32x32x8 GEMM test - WITH PERMUTATION AND INVERSE WRITEBACK
M = 32
N = 32
K = 8
WARP_SIZE = 64

def permute_row(row: int) -> int:
    """Forward permutation applied to input A rows."""
    high = (row >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    return (row & 0x3) | (rotated << 2)

def inverse_permute_row(permuted: int) -> int:
    """Inverse permutation - given a permuted row, return the original row."""
    # We need to find what original row maps to this permuted row
    for orig in range(32):
        if permute_row(orig) == permuted:
            return orig
    return -1

# Pre-compute inverse permutation table
INVERSE_PERM = [inverse_permute_row(p) for p in range(32)]
print(f"Inverse permutation table: {INVERSE_PERM}")

@substrate.jit
def gemm_kernel(
    A: S.Tensor((32, 8), S.bf16),
    B: S.Tensor((8, 32), S.bf16),
    C: S.Tensor((32, 32), S.f32),
):
    lane = S.thread_id(0)
    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    acc = S.full((16,), 0.0, S.f32)

    # Row permutation for A operand
    high = (lane_col >> 2) & 0x7
    rotated = ((high & 0x1) << 2) | (high >> 1)
    permuted_row = (lane_col & 0x3) | (rotated << 2)

    a_row = permuted_row  # Load from permuted row
    b_col = lane_col

    # Load 4 bf16 values into local buffers
    a_frag_local = S.make_local((4,), S.bf16)
    b_frag_local = S.make_local((4,), S.bf16)

    for e in S.range(4):
        k = lane_k_base + e
        a_frag_local[e] = A[a_row, k]
        b_frag_local[e] = B[k, b_col]

    acc = S.amdgpu.mfma_32x32x8_bf16_f32(a_frag_local, b_frag_local, acc)

    # Writeback with INVERSE permutation
    # The MFMA output layout: row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)
    # This is a "logical row" in the MFMA's internal coordinate system
    # We loaded from permuted_row, so this logical row corresponds to permuted_row
    # We need to write back to the original (un-permuted) row
    for acc_idx in S.range(16):
        # The logical row in MFMA output
        logical_row = 8 * (acc_idx // 4) + 4 * (lane // 32) + (acc_idx % 4)

        # We loaded from permuted_row, and the MFMA treats this as row 0 of its computation
        # So logical_row 0 corresponds to permuted_row
        # We need to write to the original row that permutes to (permuted_row + logical_row)

        # Actually, MFMA computes 32x32 output. Each lane holds 16 elements.
        # The row assignment in the output depends on the input permutation.

        col = lane_col
        # Write directly - the permutation is in the load, output should match
        C[logical_row, col] = acc[acc_idx]


if __name__ == "__main__":
    # Test - we need to pre-permute A to match what the kernel expects
    A = torch.randn((32, 8), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((8, 32), dtype=torch.bfloat16, device='cuda')
    C = torch.zeros((32, 32), dtype=torch.float32, device='cuda')

    # The kernel loads A with permutation, so we need A to be pre-permuted
    # OR we need to adjust our thinking...

    # Actually, let's think about this differently:
    # The kernel loads A[permuted_row, k] for each lane
    # So lane_col 0 loads from row permute_row(0), lane_col 1 from permute_row(1), etc.
    # This means the data in A should be at the original positions,
    # and the kernel reads it with permutation to get the correct swizzle for MFMA

    expected = (A.float() @ B.float())

    gemm_kernel[lambda: ((1, 1, 1), (WARP_SIZE, 1, 1))](A, B, C)

    actual = C.cpu()
    expected_cpu = expected.cpu()

    max_diff = torch.max(torch.abs(actual - expected_cpu)).item()
    print(f'Max diff: {max_diff}')
    print(f'Pass: {torch.allclose(actual, expected_cpu, rtol=1e-2, atol=0.1)}')

    # Check which rows match
    for r in range(0, 32, 4):
        row_diff = torch.max(torch.abs(actual[r] - expected_cpu[r])).item()
        match = "MATCH" if row_diff < 0.01 else "MISMATCH"
        print(f'Row {r} diff: {row_diff} {match}')
