#!/usr/bin/env python3
"""Debug test to understand MFMA indexing."""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# Create simple inputs
x = torch.randn((1, 1, 20, 20), dtype=torch.float16, device='cuda')
w = torch.randn((1, 1, 3, 3), dtype=torch.float16, device='cuda')

print("Weight matrix:")
print(w[0, 0])

# Compute reference
ref_full = F.conv2d(x.float(), w.float()).half()
ref = ref_full[:, :, :16, :16]

print("\nReference Y[0,0,0,:4]:", ref[0, 0, 0, :4])

# Manually compute Y[0][1] to verify
y01_manual = 0.0
for k0 in range(3):
    for k1 in range(3):
        y01_manual += x[0, 0, 0 + k0, 1 + k1].float().item() * w[0, 0, k0, k1].float().item()

print(f"\nManual Y[0][1] = {y01_manual}")
print(f"Reference Y[0][1] = {ref[0, 0, 0, 1].float().item()}")

# Check input values
print("\nInput values used for Y[0][1]:")
for k0 in range(3):
    for k1 in range(3):
        i0 = 0 + k0
        i1 = 1 + k1
        val = x[0, 0, i0, i1].float().item()
        wt = w[0, 0, k0, k1].float().item()
        print(f"  X[{i0},{i1}] = {val}, W[{k0},{k1}] = {wt}, product = {val * wt}")

# Check what the expected A[0][1] should be for each kernel position
print("\nExpected A[0][1] for each kernel position:")
for k0 in range(3):
    for k1 in range(3):
        i0 = 0 + k0
        i1 = 1 + k1
        a_val = x[0, 0, i0, i1].float().item()
        print(f"  k0={k0}, k1={k1}: A[0][1] = X[{i0},{i1}] = {a_val}")
