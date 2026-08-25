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


# ---------------------------------------------------------------------------
# Low-discrepancy sequences
#
# A low-discrepancy sequence is a construction in s DIMENSIONS: point i is a
# vector whose j-th coordinate comes from a different one-dimensional
# construction (a different prime base for Halton, a different set of direction
# numbers for Sobol). The whole point is that the resulting POINT SET fills the
# s-dimensional unit cube more evenly than independent random draws.
#
# The weight matrix is therefore filled ROW BY ROW: row i is one s-dimensional
# point, where s = fan-in. Each neuron's incoming weight vector is a single
# sample, and the sequence stratifies the fan-in space across neurons.
#
# Filling the matrix from a SINGLE one-dimensional sequence read left-to-right
# instead (the original implementation here) is not merely a weaker version of
# this: with 65,536 = 2^16 values from a base-2 construction laid out as
# (256, 256), every row becomes an affine copy of the same pattern and the dense
# matrix has effective rank 1. The sparse mask hides most of the damage — the
# masked matrix recovers to rank ~171 against Gaussian's ~203 — but the
# construction still does not match what the Background chapter describes, so
# any result attributed to "low-discrepancy initialization" would be measuring
# something else.
#
# Both generators below were cross-checked against John Burkardt's reference
# implementations (people.sc.fsu.edu/~jburkardt/py_src/{halton,sobol}); see
# `scripts/check_lds_against_burkardt.py`.
# ---------------------------------------------------------------------------

# Sobol's point 0 is the origin and its point 1 is the all-0.5 vector. Under the
# [0,1] -> [-1,1] shift those become an all-(-1) row and an exactly ZERO row: one
# saturated neuron and one with no recurrent input at all. Both are artefacts of
# where the sequence starts rather than properties of the sequence, so the first
# two points are skipped. Skipping only the origin is NOT enough.
_SOBOL_SKIP = 2


def _point_layout(shape):
    """Split `shape` into (number of points, dimensions per point).

    The last axis is the fan-in, so it becomes the dimension of the sequence and
    every other axis enumerates points.
    """
    dimension = shape[-1] if len(shape) > 1 else 1
    return math.prod(shape) // dimension, dimension


def generate_sobol_weights(shape, scale=1.0):
    """
    Weight tensor from an s-dimensional Sobol sequence, one point per row.

    Uses `torch.quasirandom.SobolEngine`, which carries the Joe-Kuo (2008)
    direction numbers. Burkardt's reference port implements Bratley-Fox
    (ACM Algorithm 659, 1988) instead: the two agree exactly in dimensions 1-2
    and diverge from dimension 3 onwards, where the initial direction numbers
    first become a free choice. Both are valid Sobol sequences and their
    one-dimensional uniformity is identical; the reference caps at 40 dimensions
    and so cannot serve as the generator here, only as an oracle.
    """
    num_points, dimension = _point_layout(shape)

    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=False)
    points = engine.draw(num_points + _SOBOL_SKIP)[_SOBOL_SKIP:]

    return ((points * 2.0 - 1.0) * scale).to(torch.float32).view(shape)


def _first_primes(count):
    """The first `count` primes, as Halton's per-dimension bases."""
    if count < 1:
        return []
    # Overestimate of the count-th prime (n ln n + n ln ln n), then sieve.
    limit = 16 if count < 6 else int(count * (math.log(count) + math.log(math.log(count)))) + 10
    while True:
        sieve = bytearray([1]) * (limit + 1)
        sieve[0:2] = b"\x00\x00"
        for p in range(2, int(limit ** 0.5) + 1):
            if sieve[p]:
                sieve[p * p::p] = bytearray(len(range(p * p, limit + 1, p)))
        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def _next_prime_above(value):
    """Smallest prime strictly greater than `value`."""
    candidate = value + 1
    while True:
        if candidate > 1 and all(candidate % d for d in range(2, int(candidate ** 0.5) + 1)):
            return candidate
        candidate += 1


def _radical_inverse(indices, base):
    """Radical inverse of each index in `base`, vectorised over the point index.

    Identical arithmetic to the textbook scalar loop (and bit-identical to
    Burkardt's `halton()`), evaluated for all points at once.
    """
    remaining = indices.clone()
    result = torch.zeros(len(remaining), dtype=torch.float64)
    fraction = 1.0
    while bool((remaining > 0).any()):
        fraction /= base
        result += fraction * (remaining % base).to(torch.float64)
        remaining = remaining // base
    return result


def generate_halton_weights(shape, scale=1.0, leap=None):
    """
    Weight tensor from an s-dimensional Halton sequence, one point per row.

    Dimension j uses the j-th prime as its base, as in Burkardt's reference
    `halton(i, m)` — which this reproduces exactly.

    `leap` addresses the well-known high-dimensional Halton degeneracy. With 256
    dimensions the largest base is the 256th prime, 1619, and the radical inverse
    of i < 256 in base 1619 is simply i/1619 — a linear ramp. Enough dimensions
    behave that way that the dense point set has effective rank ~26 rather than
    the ~200 of a Gaussian draw. Taking every `leap`-th point pushes the indices
    past the first digit of the large bases and repairs it (dense rank ~189,
    masked ~204, on par with Gaussian).

    Following Kocis and Whiten (1997), the leap must be a prime that is NOT one
    of the bases in use; the default is therefore the smallest prime above the
    largest base, which satisfies that by construction at any dimension rather
    than hard-coding a constant that happens to work at 256.

    Leaping is preferred over scrambling because it keeps the sequence
    deterministic: Sobol and Halton then retain exactly zero seed-to-seed
    variance (see `DETERMINISTIC_METHODS`), which is a reported property here,
    whereas scrambling would introduce a seed and change that claim.
    """
    num_points, dimension = _point_layout(shape)

    bases = _first_primes(dimension)
    if leap is None:
        leap = _next_prime_above(bases[-1])

    indices = torch.arange(1, num_points + 1, dtype=torch.long) * leap
    columns = [_radical_inverse(indices, base) for base in bases]
    points = torch.stack(columns, dim=1)

    return ((points * 2.0 - 1.0) * scale).to(torch.float32).view(shape)


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