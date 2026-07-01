import torch
import substrate
import substrate.language as S

# Test to verify loading from different rows

M = 64
K = 8
WARP_SIZE = 64

@substrate.jit
def load_test_kernel(
    A: S.Tensor((64, 8), S.f32),
    out: S.Tensor((64, 8), S.f32),  # Output to verify loading
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Load from row lane_col
    for e in S.range(4):
        k_idx = lane_k_base + e
        out[lane, k_idx] = A[lane_col, e]

    # Also write lane_col to verify
    out[lane, 4] = lane_col


if __name__ == "__main__":
    A = torch.randn((64, 8), dtype=torch.float32, device='cuda')
    out = torch.zeros((64, 8), dtype=torch.float32, device='cuda')

    load_test_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, out)

    out_cpu = out.cpu()
    A_cpu = A.cpu()

    print("Verifying loading pattern:")
    print(f"lane 0: lane_col = {out_cpu[0, 4].item()}, loaded A[{out_cpu[0, 4].item()}, :4] = {out_cpu[0, :4]}")
    print(f"         A[0, :4] = {A_cpu[0, :4]}")

    print(f"lane 31: lane_col = {out_cpu[31, 4].item()}, loaded A[{out_cpu[31, 4].item()}, :4] = {out_cpu[31, :4]}")
    print(f"         A[31, :4] = {A_cpu[31, :4]}")

    print(f"lane 32: lane_col = {out_cpu[32, 4].item()}, loaded A[{out_cpu[32, 4].item()}, :4] = {out_cpu[32, :4]}")
    print(f"         A[0, :4] = {A_cpu[0, :4]}")

    print(f"lane 63: lane_col = {out_cpu[63, 4].item()}, loaded A[{out_cpu[63, 4].item()}, :4] = {out_cpu[63, :4]}")
    print(f"         A[31, :4] = {A_cpu[31, :4]}")


# Now test loading from shifted rows
@substrate.jit
def load_shifted_kernel(
    A: S.Tensor((64, 8), S.f32),
    out: S.Tensor((64, 8), S.f32),
):
    tid = S.thread_id(0)
    lane = tid % 64

    lane_col = lane % 32
    lane_k_base = (lane // 32) * 4

    # Load from row lane_col + 32
    for e in S.range(4):
        k_idx = lane_k_base + e
        out[lane, k_idx] = A[lane_col + 32, e]

    out[lane, 4] = lane_col + 32


print("\n\nVerifying shifted loading pattern:")
out2 = torch.zeros((64, 8), dtype=torch.float32, device='cuda')
load_shifted_kernel[lambda: ((1, 1, 1), (64, 1, 1))](A, out2)

out2_cpu = out2.cpu()
print(f"lane 0: loaded A[{out2_cpu[0, 4].item()}, :4] = {out2_cpu[0, :4]}")
print(f"        A[32, :4] = {A_cpu[32, :4]}")

print(f"lane 31: loaded A[{out2_cpu[31, 4].item()}, :4] = {out2_cpu[31, :4]}")
print(f"         A[63, :4] = {A_cpu[63, :4]}")

print(f"lane 32: loaded A[{out2_cpu[32, 4].item()}, :4] = {out2_cpu[32, :4]}")
print(f"        A[32, :4] = {A_cpu[32, :4]}")
