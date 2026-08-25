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
  temporal_analysis.py  per-timestep separability + prefix-classifier curves
  training.py           shared truncated-BPTT trainer (weighted loss, early stop)
  tuning.py             hyperparameter search (exhaustive and staged)
  evaluation_metrics.py Macro-F1, confusion matrices, latency, SOPs/MACs energy
  models/
    spiking_reservoir.py  LIF neurons + surrogate gradient + k-truncation
    lstm.py               dense LSTM baseline
    lsm.py                Liquid State Machine (frozen reservoir, trained readout)
  experiments.py        model factory, run bookkeeping, resumable result store
notebooks/              01-EDA, 02-temporal-analysis, 03-baselines,
                        04-initialization-study, 05-truncation-study
scripts/                figure generation (e.g. LIF neuron dynamics)
results/                run outputs (CSV/JSON)
Data/                   UCR datasets
```

## Setup

Dependencies and the `edgesnn` package are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                # install dependencies + editable package into .venv
uv run jupyter lab     # launch notebooks (start with notebooks/01-exploratory-data-analysis.ipynb)
```

Modules are then importable anywhere: `from edgesnn.data_utils import parse_ucr_file`.

## Status

- [x] Data loading + EDA (parsing, binary parse cache, class-distribution and mean/std plots)
- [x] Cheap initializers (Gaussian baseline, LCG, LFSR, Sobol, Halton) with sparse
      reservoir connectivity (`p_c`) and seeding
- [x] Spiking reservoir model (LIF + surrogate gradient + `k`-truncation, spike-based
      recurrence, structurally enforced sparsity)
- [x] Dataset class + DataLoaders (label encoding, normalization)
- [x] Dense LSTM baseline (`models/lstm.py`), parameter-matched to the reservoir
      and sharing its truncation window
- [x] Temporal analysis (`temporal_analysis.py`): per-timestep ANOVA separability
      and cross-validated prefix-classifier curves, computed on the training split
- [x] Liquid State Machine baseline (`models/lsm.py`): the same spiking substrate
      with a frozen reservoir and a trained linear readout
- [x] Truncated surrogate-gradient BPTT trainer (`training.py`) shared by all three
      models, with class-weighted loss, a stratified validation split and early
      stopping on Macro-F1
- [x] Exhaustive hyperparameter search (`tuning.py`) over learning rate, four
      surrogate-gradient families and the neuron parameters, selected by
      cross-validation on the training split
- [x] Evaluation metrics (`evaluation_metrics.py`): Macro-F1, confusion matrices,
      algorithmic latency, and SOPs-vs-MACs energy under the Horowitz model
- [x] Experiment notebooks for the three staged comparisons (`03-baselines`,
      `04-initialization-study`, `05-truncation-study`), resumable across restarts
- [x] Low-discrepancy initializers reworked to the s-dimensional construction
      (one sequence point per row, one prime base per input dimension), with a
      leaped Halton to avoid the high-dimensional degeneracy and the first two
      Sobol points skipped; validated against an independent reference
      implementation by `scripts/check_lds_against_burkardt.py`
- [x] Stage 1: all five initializers on a frozen reservoir over five seeds each,
      with the weight-variance control ablated
- [x] Stages 2 and 3: the truncation sweep across `k`, with operation counts,
      energy estimates and the accuracy-versus-energy trade-off
- [x] Gradient clipping applied uniformly to all three models, so that the dense
      baseline is not the only one exposed to gradient explosion, with the LSTM's
      learning-rate grid widened to match the search effort the other models
      receive
- [x] A single epoch budget shared by every model and stage, set high enough to
      act as a safety ceiling rather than a training length, so that runs
      terminate on the early-stopping criterion instead of on the budget; whether
      the ceiling was reached is recorded for every run
- [ ] Final run of the full pipeline across all three datasets

### Reproducing the results

Run the notebooks in order; each writes to `reports/results/` and skips work already done,
so an interrupted sweep resumes rather than restarting.

```bash
uv run jupyter lab
# 01-EDA                  dataset figures and Table 3.1
# 02-temporal-analysis    per-timestep separability, prefix-classifier curves
# 03-baselines            hyperparameter search  -> results/config_*.json
# 04-initialization-study Stage 1: the five initializers on a frozen reservoir
# 05-truncation-study     Stages 2 and 3: the k sweep, energy and latency
```

Notebook 03 must run first: it selects the configurations that 04 and 05 reuse
unchanged. Both raise a clear error if those configurations are missing.

### Model configuration

The spiking reservoir uses `N_res = 256` neurons at a connection probability of
`p_c = 0.1`, and the LSTM baseline uses a hidden size of `H = 128`. The pairing is
chosen so the two models carry an equal trainable-parameter budget: an LSTM
allocates four gate matrices per hidden unit, costing roughly `4H²` recurrent
parameters against the reservoir's `N_res²`, so parity falls at `H ≈ N_res / 2`.
Both land at ~66–68k parameters. Sparsity is deliberately excluded from that
matching, so the efficiency benefit appears in the operation counts rather than
in the choice of layer size. Both models are evaluated over the same truncation
grid, `k ∈ {20, 40, 60, 80, 100}%` of sequence length.
