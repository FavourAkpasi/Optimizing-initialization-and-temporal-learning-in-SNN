# Optimizing Initialization and Temporal Learning for Edge Spiking Neural Networks

Master thesis · **Favour Akpasi** · Institute of Signal Processing (ISP), JKU Linz

## Overview

This project builds a **co-optimized Spiking Recurrent Neural Network (SNN)** for
edge hardware, aimed at tackling two computational bottlenecks at once:

1. **Initialization cost.** Standard networks draw weights from a Gaussian
   distribution (Box–Muller: logs, square roots) — expensive on tiny
   microcontrollers. Can mathematically *cheap* initializers — uniform PRNGs
   (LCG, LFSR) or low-discrepancy sequences (Sobol, Halton) — replace Gaussian
   randomness without reducing performance?
2. **Latency / energy cost.** Recurrent networks process every timestep. By
   *truncating* learning and inference at timestep `k` (via surrogate-gradient
   descent), the network can classify early and let the hardware power down
   sooner — cutting latency and dynamic energy (measured in SOPs, not MACs).

**The hypothesis:** a few epochs of truncated learning coupled with an inexpensive
hardware initialization, could provide an architecture that can match a standard
LSTM at a fraction of the energy.

## Experimental design

Three staged comparisons:

1. **Init vs. init** — frozen reservoir, cheap PRNG/LDS weights vs. Gaussian
   baseline. *How much accuracy do we lose with cheap math?*
2. **Preset vs. learned** — best cheap init + truncated learning vs. frozen.
   *Can truncated learning fix a cheap init?*
3. **Edge viability** — our hybrid vs. a dense LSTM and a random LSM, on
   Macro-F1, SOPs/MACs, and latency.

**Datasets** (UCR archive, in `Data/`): ECG5000, Wafer, InsectSound.

## Project structure

```
src/edgesnn/            installable package (editable via `uv sync`)
  data_utils.py         UCR parsing/loading (format-agnostic .txt/.tsv/.ts)
  plotting.py           EDA plots (operate on DataFrames)
  initializers.py       cheap PRNG/LDS weight initializers
  models/
    spiking_reservoir.py  LIF neurons + surrogate gradient + k-truncation
notebooks/              experiments (currently: 01-EDA.ipynb)
Data/                   UCR datasets
```

## Setup

Dependencies and the `edgesnn` package are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                # install dependencies + editable package into .venv
uv run jupyter lab     # launch notebooks (start with notebooks/01-EDA.ipynb)
```

Modules are then importable anywhere: `from edgesnn.data_utils import parse_ucr_file`.

## Status

- [x] Data loading + EDA (parsing, binary parse cache, class-distribution and mean/std plots)
- [x] Cheap initializers (Gaussian baseline, LCG, LFSR, Sobol, Halton) with sparse
      reservoir connectivity (`p_c`) and seeding
- [x] Spiking reservoir model (LIF + surrogate gradient + `k`-truncation, spike-based
      recurrence, structurally enforced sparsity)
- [x] Dataset class + DataLoaders (label encoding, normalization)
- [ ] Baselines: dense LSTM, random LSM
- [ ] Truncated surrogate-gradient BPTT trainer
- [ ] Evaluation metrics (Macro-F1, confusion matrix, latency, SOPs vs. MACs)
- [ ] Experiment notebooks for the three staged comparisons
