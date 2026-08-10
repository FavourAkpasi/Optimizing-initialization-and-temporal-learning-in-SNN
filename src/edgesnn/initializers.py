import torch
import math


def generate_lcg_weights(shape, seed=12345, scale=1.0):
    """
    Generates a weight tensor using a Linear Congruential Generator (LCG).
    This simulates a cheap, sequential PRNG common on edge hardware.
    Formula: X_{n+1} = (a * X_n + c) mod m
    """
    num_elements = math.prod(shape)

    # Standard POSIX.1-2001 LCG parameters
    m = 2 ** 31
    a = 1103515245
    c = 12345

    state = seed
    values = []

    for _ in range(num_elements):
        state = (a * state + c) % m
        # Normalize to [0, 1] then shift to [-1, 1]
        normalized = (state / m) * 2.0 - 1.0
        values.append(normalized * scale)

    tensor = torch.tensor(values, dtype=torch.float32).view(shape)
    return tensor


def generate_lfsr_weights(shape, seed=0xACE1, scale=1.0):
    """
    Generates a weight tensor using a 16-bit Galois Linear Feedback Shift Register (LFSR).
    This simulates highly hardware-efficient bitwise randomness.
    """
    num_elements = math.prod(shape)
    state = seed
    values = []

    for _ in range(num_elements):
        # 16-bit LFSR taps: 16, 14, 13, 11
        lsb = state & 1
        state >>= 1
        if lsb == 1:
            state ^= 0xB400

        # Normalize to [-1, 1] (Max value for 16-bit is 65535)
        normalized = (state / 65535.0) * 2.0 - 1.0
        values.append(normalized * scale)

    tensor = torch.tensor(values, dtype=torch.float32).view(shape)
    return tensor


def generate_sobol_weights(shape, scale=1.0):
    """
    Generates a weight tensor using the Sobol Low-Discrepancy Sequence.
    Utilizes PyTorch's built-in Quasi-Random sequence engine for maximum uniformity.
    """
    num_elements = math.prod(shape)

    # We use a 1D Sobol engine and draw enough numbers to fill the tensor
    sobol_engine = torch.quasirandom.SobolEngine(dimension=1)

    # Draw values in [0, 1]
    raw_values = sobol_engine.draw(num_elements).squeeze()

    # Shift to [-1, 1] and scale (typically by 1/sqrt(fan_in) like Kaiming/Xavier)
    scaled_values = (raw_values * 2.0 - 1.0) * scale

    return scaled_values.view(shape)


def generate_halton_weights(shape, scale=1.0, base=2):
    """
    Generates a weight tensor using a 1D Halton Sequence.
    Mathematically constructs a deterministic sequence using radical inverse functions.
    """
    num_elements = math.prod(shape)
    values = []

    for i in range(1, num_elements + 1):
        f = 1.0
        r = 0.0
        index = i
        while index > 0:
            f = f / base
            r = r + f * (index % base)
            index = index // base

        # r is in [0, 1]. Shift to [-1, 1]
        normalized = r * 2.0 - 1.0
        values.append(normalized * scale)

    tensor = torch.tensor(values, dtype=torch.float32).view(shape)
    return tensor


def initialize_reservoir(shape, method='sobol', fan_in=None):
    """
    Factory function to initialize reservoir weights.
    Applies standard fan-in scaling to keep variance stable.
    """
    if fan_in is None:
        fan_in = shape[1] if len(shape) > 1 else shape[0]

    scale = 1.0 / math.sqrt(fan_in)

    if method == 'gaussian':
        return torch.randn(shape) * scale
    elif method == 'lcg':
        return generate_lcg_weights(shape, scale=scale)
    elif method == 'lfsr':
        return generate_lfsr_weights(shape, scale=scale)
    elif method == 'sobol':
        return generate_sobol_weights(shape, scale=scale)
    elif method == 'halton':
        return generate_halton_weights(shape, scale=scale)
    else:
        raise ValueError(f"Unknown initialization method: {method}")