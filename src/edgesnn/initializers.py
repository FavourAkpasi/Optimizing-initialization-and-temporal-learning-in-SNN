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


# Methods whose raw output is uniform on [-1, 1]. A uniform variate on [-1, 1]
# has standard deviation 1/sqrt(3), whereas `torch.randn` has standard deviation
# 1. Multiplying the uniform generators by sqrt(3) equalises the two.
_UNIFORM_METHODS = frozenset({"lcg", "lfsr", "sobol", "halton"})
_UNIFORM_VARIANCE_CORRECTION = math.sqrt(3.0)

# Sobol and Halton are deterministic constructions: they produce the same points
# on every call, so a "seed" has nothing to act on. This is a property of
# low-discrepancy sequences rather than a limitation of the implementation, and
# it means these methods have exactly zero seed-to-seed variance. Experiment
# runners should consult this set instead of silently repeating identical runs.
DETERMINISTIC_METHODS = frozenset({"sobol", "halton"})


def is_deterministic(method):
    """True if `method` ignores the seed and yields identical weights every run."""
    return method in DETERMINISTIC_METHODS


def generate_sparse_mask(shape, connectivity, seed=None):
    """
    Binary connectivity mask with an EXACT number of non-zero entries.

    Reservoir computing keeps the recurrent matrix sparse (a connection
    probability p_c around 0.1) so that each neuron drives only a few
    postsynaptic partners. Here sparsity is not merely convention: the number of
    non-zero entries in W_res is precisely the recurrent fan-out that a spike
    must pay for, so it is the term that converts a spike count into Synaptic
    Operations (SOPs).

    An exact count (rather than an i.i.d. Bernoulli draw per entry) is used so
    that every initializer, seed and dataset shares the identical synaptic
    budget. Otherwise two runs would differ in energy for a reason that has
    nothing to do with the initializer under study.
    """
    if not 0.0 < connectivity <= 1.0:
        raise ValueError(f"connectivity must be in (0, 1], got {connectivity}")

    numel = math.prod(shape)
    num_active = max(1, int(round(connectivity * numel)))

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))

    # randperm picks `num_active` distinct positions -> exact, uniform, no retry loop.
    idx = torch.randperm(numel, generator=generator)[:num_active]
    mask = torch.zeros(numel, dtype=torch.float32)
    mask[idx] = 1.0
    return mask.view(shape)


def initialize_reservoir(shape, method='sobol', fan_in=None, connectivity=None,
                         seed=None, match_variance=True):
    """
    Factory function to initialize reservoir weights.
    Applies standard fan-in scaling to keep variance stable.

    `connectivity` (p_c) optionally applies a sparse mask. When it is set, the
    fan-in used for scaling is the EFFECTIVE fan-in (p_c * fan_in), i.e. the
    number of synapses that actually reach a neuron. Scaling by the dense fan-in
    instead would shrink the variance of the recurrent current by a factor p_c
    and leave the reservoir almost silent — too few spikes to carry information,
    which would confound a study whose dependent variable is the spike count.

    `match_variance=True` (the default) equalises the weight variance across all
    methods, so that every initializer starts from Var[w] = 1/effective_fan_in.
    This is an experimental control, not cosmetics: `torch.randn` has standard
    deviation 1, while a uniform variate on [-1, 1] has 1/sqrt(3), so naive
    shared scaling would hand the Gaussian baseline weights ~1.73x larger than
    every cheap generator. Any accuracy gap would then confound the *shape* of
    the distribution — the actual object of study — with its *scale*. Set it to
    False only to reproduce that confound deliberately as an ablation.

    `seed` controls the stochastic parts: the Gaussian draw and the sparsity
    pattern. Sobol and Halton ignore it by construction (see
    `DETERMINISTIC_METHODS`).
    """
    if fan_in is None:
        fan_in = shape[1] if len(shape) > 1 else shape[0]

    effective_fan_in = fan_in if connectivity is None else max(1.0, connectivity * fan_in)
    scale = 1.0 / math.sqrt(effective_fan_in)

    if match_variance and method in _UNIFORM_METHODS:
        scale *= _UNIFORM_VARIANCE_CORRECTION

    if method == 'gaussian':
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        weights = torch.randn(shape, generator=generator) * scale
    elif method == 'lcg':
        weights = generate_lcg_weights(shape, scale=scale,
                                       **({} if seed is None else {"seed": int(seed)}))
    elif method == 'lfsr':
        # A zero state is the LFSR's absorbing state (it would emit a constant),
        # so a zero seed is shifted to the standard non-zero default.
        lfsr_kwargs = {} if seed is None else {"seed": int(seed) & 0xFFFF or 0xACE1}
        weights = generate_lfsr_weights(shape, scale=scale, **lfsr_kwargs)
    elif method == 'sobol':
        weights = generate_sobol_weights(shape, scale=scale)
    elif method == 'halton':
        weights = generate_halton_weights(shape, scale=scale)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    if connectivity is not None:
        weights = weights * generate_sparse_mask(shape, connectivity, seed=seed)

    return weights