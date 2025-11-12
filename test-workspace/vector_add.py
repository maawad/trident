import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # pointer to the input vector x
    y_ptr,  # pointer to the input vector y
    output_ptr,  # pointer to the output vector
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)

    # Calculate the starting index
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to guard memory operations
    mask = offsets < n_elements

    # Load x and y from DRAM, masking out any extra elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Add x and y
    output = x + y

    # Write back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    """Vector addition using Triton"""
    # Allocate output tensor
    output = torch.empty_like(x)

    # Ensure contiguous
    assert x.is_contiguous() and y.is_contiguous()

    n_elements = output.numel()

    # Launch kernel with 1D grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    # Test the kernel
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    # Triton
    output_triton = add(x, y)

    # PyTorch
    output_torch = x + y

    # Verify
    print(f"Triton and PyTorch match: {torch.allclose(output_triton, output_torch)}")
    print(f"Max difference: {torch.max(torch.abs(output_triton - output_torch))}")

