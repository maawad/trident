# Triton Test Workspace

This directory contains sample Triton kernels for testing the assembly viewer extension.

## Files

- `vector_add.py`: Simple vector addition kernel
- `matmul.py`: Matrix multiplication kernel

## Usage

1. Load PyTorch module:
   ```bash
   module load pytorch
   ```

2. Run a kernel to generate cache:
   ```bash
   python vector_add.py
   ```

3. Open the Python file in VS Code and click the split view icon to view assembly

## Expected Output

After running, assembly files will be cached in `~/.triton/cache` and can be viewed through the extension.

