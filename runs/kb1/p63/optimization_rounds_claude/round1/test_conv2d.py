import torch
import torch.nn as nn
from output_model_new import ModelNew

# Test the Conv2D kernel
N = 16
IN_CHANNELS = 16
IN_H = 1024
IN_W = 1024
OUT_CHANNELS = 128
KERNEL_H = 3
KERNEL_W = 3
OUT_H = 1022
OUT_W = 1022

# Create model
model = ModelNew(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    kernel_size=(KERNEL_H, KERNEL_W),
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
)

# Create input
x = torch.randn(N, IN_CHANNELS, IN_H, IN_W, dtype=torch.float32, device='cuda')

# Run forward pass
print("Running forward pass...")
y = model(x)

# Run reference
print("Running reference...")
ref_conv = nn.Conv2d(
    IN_CHANNELS, OUT_CHANNELS, (KERNEL_H, KERNEL_W),
    stride=1, padding=0, dilation=1, groups=1, bias=False
)
ref_conv.weight.data = model.conv2d.weight.data.clone()
ref_conv = ref_conv.cuda()
y_ref = ref_conv(x)

# Compare
print("Comparing results...")
max_diff = torch.max(torch.abs(y - y_ref)).item()
mean_diff = torch.mean(torch.abs(y - y_ref)).item()

print(f"Max diff: {max_diff}")
print(f"Mean diff: {mean_diff}")
print(f"Pass: {torch.allclose(y, y_ref, rtol=1e-2, atol=0.1)}")

# Check output shape
print(f"Output shape: {y.shape}")
print(f"Expected shape: {y_ref.shape}")
