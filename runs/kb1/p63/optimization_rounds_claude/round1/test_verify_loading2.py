import torch
import substrate
import substrate.language as S

# Test to verify loading from different rows - fixed version

M = 64
K = 8
WARP_SIZE = 64

@substrate.jit
def load_test_kernel(
    A: S.Tensor((64, 8), S.f32),
    out: S.Tensor((64, 4), S.f32),  # Output to verify loading
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Load from row lane_col, write to out[lane, :]
    for e in S.range(4):
        out[lane, e] = A[lane_col, e]


@substrate.jit
def load_shifted_kernel(
    A: S.Tensor((64, 8), S.f32),
    out: S.Tensor((64, 4), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32

    # Load from row lane_col + 32, write to out[lane, :]
    for e in S.range(4):
        out[lane, e] = A[lane_col + 32, e]


if __name__ == "__main__":
    A = torch.randn((64, 8), dtype=torch.float32, device='cuda')
    out = torch.zeros((64, 4), dtype=torch.float32, device='cuda')

    load_test_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, out)

    out_cpu = out.cpu()
    A_cpu = A.cpu()

    print("Verifying loading pattern (from row lane_col):")
    for lane in [0, 15, 16, 31, 32, 47, 48, 63]:
        lane_col = lane % 32
        print(f"  lane {lane}: lane_col={lane_col}, out={out_cpu[lane].tolist()}")
        print(f"           A[{lane_col}, :4] = {A_cpu[lane_col, :4].tolist()}")
        match = torch.allclose(out_cpu[lane], A_cpu[lane_col, :4])
        print(f"           Match: {match}")

    print("\n\nVerifying shifted loading pattern (from row lane_col + 32):")
    out2 = torch.zeros((64, 4), dtype=torch.float32, device='cuda')
    load_shifted_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, out2)

    out2_cpu = out2.cpu()
    for lane in [0, 15, 16, 31, 32, 47, 48, 63]:
        lane_col = lane % 32
        row = lane_col + 32
        print(f"  lane {lane}: loading from row {row}, out={out2_cpu[lane].tolist()}")
        print(f"           A[{row}, :4] = {A_cpu[row, :4].tolist()}")
        match = torch.allclose(out2_cpu[lane], A_cpu[row, :4])
        print(f"           Match: {match}")
