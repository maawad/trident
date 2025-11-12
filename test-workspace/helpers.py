"""
Helper functions for Triton kernels
"""

import triton
import triton.language as tl


@triton.jit
def clamp(x, min_val, max_val):
    """Clamp a value between min and max"""
    return tl.maximum(tl.minimum(x, max_val), min_val)


@triton.jit
def relu(x):
    """ReLU activation function"""
    return tl.maximum(x, 0.0)


@triton.jit
def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + tl.exp(-x))

