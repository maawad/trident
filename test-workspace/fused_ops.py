"""
Fused operations using helper functions from helpers.py
"""

import torch
import triton
import triton.language as tl
from helpers import relu, sigmoid, clamp


@triton.jit
def fused_matmul_relu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused matrix multiplication with ReLU activation
    Demonstrates using helper functions from another file
    """
    # Get program IDs
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Do the matrix multiplication
        accumulator += tl.dot(a, b)

        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply ReLU from helpers.py
    c = relu(accumulator)

    # Convert to output dtype
    c = c.to(tl.float16)

    # Write back the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def normalize_clamp_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize values and clamp to range using helper function
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Normalize (simple mean normalization for demo)
    mean = tl.sum(x) / tl.sum(mask.to(tl.float32))
    normalized = x - mean

    # Apply clamp from helpers.py
    clamped = clamp(normalized, min_val, max_val)

    # Store result
    tl.store(output_ptr + offsets, clamped, mask=mask)


def fused_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    """Fused matmul + ReLU"""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    fused_matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )

    return c


def normalize_clamp(x: torch.Tensor, min_val: float, max_val: float):
    """Normalize and clamp values"""
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    normalize_clamp_kernel[grid](x, output, n_elements, min_val, max_val, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    # Test fused matmul + relu
    M, N, K = 512, 512, 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    output = fused_matmul_relu(a, b)

    # PyTorch reference
    torch_output = torch.relu(torch.matmul(a, b))

    print(f"Fused matmul+relu matches PyTorch: {torch.allclose(output, torch_output, atol=1e-2)}")

    # Test normalize + clamp
    x = torch.randn(10000, device='cuda', dtype=torch.float32)
    clamped = normalize_clamp(x, -2.0, 2.0)

    print(f"Clamped range: [{clamped.min():.2f}, {clamped.max():.2f}]")
    print(f"All values in range: {(clamped >= -2.0).all() and (clamped <= 2.0).all()}")

