"""Classification, latency and energy metrics, implementing Methods Section 3.7.

The formulas here are fixed by the thesis text rather than chosen freely: the
Methods chapter states them as equations, and this module exists to compute
exactly those. Where a choice remains, the comment records which sentence of the
chapter settles it.

The central asymmetry is that the dense baseline's cost is *analytic* while the
spiking models' cost is *measured*. An LSTM performs the same multiply-accumulate
on every input regardless of its values, so its operation count follows from the
architecture alone. A spiking reservoir only pays for synapses that a spike
actually traverses, so its count depends on the data and must be observed. That
asymmetry is not an inconvenience of the accounting; it is the efficiency claim.

Energy uses Horowitz's 45 nm / 0.9 V figures: a 32-bit floating-point add costs
0.9 pJ and a multiply 3.7 pJ, so a multiply-accumulate costs 4.6 pJ against
0.9 pJ for the accumulate a spike requires -- a ratio of 5.1. Arithmetic only;
memory access is excluded, which makes the estimate conservative for the spiking
models, whose access pattern is the more favourable of the two.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

__all__ = [
    "E_ADD_PJ",
    "E_MULT_PJ",
    "E_MAC_PJ",
    "E_SOP_PJ",
    "classification_metrics",
    "recurrent_fanout",
    "lstm_operation_counts",
    "snn_operation_counts",
    "energy_pj",
    "evaluate_efficiency",
]

# Horowitz, ISSCC 2014, 45 nm / 0.9 V, 32-bit floating point.
E_ADD_PJ = 0.9
E_MULT_PJ = 3.7
E_MAC_PJ = E_ADD_PJ + E_MULT_PJ      # 4.6 pJ: a multiply followed by an add
E_SOP_PJ = E_ADD_PJ                  # 0.9 pJ: a spike contributes its weight unchanged


def classification_metrics(y_true, y_pred, num_classes=None):
    """Macro-F1, accuracy, per-class F1, confusion matrix and the majority floor.

    The majority-class rate is returned alongside because on Wafer a model
    predicting `normal` unconditionally scores 89.2% accuracy while detecting no
    faults at all; quoting accuracy without that floor invites the reader to
    mistake the floor for a result.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes)) if num_classes else sorted(set(y_true.tolist()))

    counts = np.bincount(y_true, minlength=len(labels))
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels,
                                 zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "majority_rate": float(counts.max() / counts.sum()),
        "support": counts.tolist(),
    }


def recurrent_fanout(model):
    """Outgoing recurrent synapses per reservoir neuron, as a `(N_res,)` tensor.

    `W_res.weight[i, j]` carries the connection from neuron `j` to neuron `i`, so
    the fan-out of `j` is the number of non-zeros in column `j`. This is the
    per-neuron `f_j` of Equation 3.6, and it is time-invariant, which is what
    lets the synaptic operation count be computed from accumulated spike counts
    rather than from the full spike raster.
    """
    cell = model.lif_layer
    mask = cell.res_mask
    if mask is None:
        return torch.full((cell.hidden_size,), float(cell.hidden_size))
    return mask.sum(dim=0).detach().cpu().float()


def lstm_operation_counts(k, hidden_size, num_classes, input_size=1):
    """Equation 3.5: `MACs = k · 4H(d + H) + C · H`. Exact and data-independent.

    The factor four is the input, forget, cell and output gates; the final term
    is the readout, applied once rather than per step.
    """
    per_step = 4 * hidden_size * (input_size + hidden_size)
    readout = num_classes * hidden_size
    return {
        "macs": float(k * per_step + readout),
        "sops": 0.0,
        "recurrent_macs": float(k * 4 * hidden_size * hidden_size),
    }


def snn_operation_counts(spike_count, fanout, k, num_classes, input_size=1):
    """Equation 3.6 plus the dense terms, per inference.

    `spike_count` is `(batch, N_res)` — spikes accumulated over the truncated
    window, exactly what `TruncatedSpikingReservoir.forward` returns. Since the
    fan-out `f_j` does not vary in time, the double sum of Equation 3.6 collapses
    to one inner product per sample.

    The input projection and the readout stay dense multiply-accumulates and are
    reported as MACs, not folded into the spiking total: `W_in x[t]` multiplies a
    real-valued input, and the readout multiplies a real-valued spike *count*.
    Counting either as a synaptic operation would overstate the saving.
    """
    spike_count = spike_count.detach().cpu().float()
    fanout = fanout.detach().cpu().float() if torch.is_tensor(fanout) else torch.as_tensor(fanout)
    n_res = spike_count.shape[1]

    sops_per_sample = (spike_count @ fanout)             # (batch,)
    input_macs = k * n_res * input_size
    readout_macs = num_classes * n_res

    return {
        "sops": float(sops_per_sample.mean()),
        "macs": float(input_macs + readout_macs),
        "input_macs": float(input_macs),
        "readout_macs": float(readout_macs),
        "spikes_per_inference": float(spike_count.sum(dim=1).mean()),
        "firing_rate": float(spike_count.sum(dim=1).mean() / (n_res * k)),
        "sops_std": float(sops_per_sample.std()),
    }


def energy_pj(macs, sops):
    """Equation 3.8: `E = 4.6 pJ · N_MAC + 0.9 pJ · N_SOP`."""
    return E_MAC_PJ * macs + E_SOP_PJ * sops


@torch.no_grad()
def evaluate_efficiency(model, dataset, k=None, batch_size=256, device="auto",
                        num_classes=None):
    """Run `model` over `dataset` and return classification, latency and energy.

    One pass produces all of it. The spike counts needed for the energy model are
    a by-product of the forward pass, so counting them separately would mean
    running the network twice to learn something the first pass already knew.

    Returns a flat dictionary suitable for a results table row.
    """
    from edgesnn.training import _with_feature_dim, resolve_device
    from torch.utils.data import DataLoader

    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()

    k = int(k if k is not None else getattr(model, "k", dataset.seq_len))
    k = min(k, dataset.seq_len)
    num_classes = num_classes or dataset.num_classes
    is_spiking = hasattr(model, "lif_layer")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, targets, spike_chunks = [], [], []

    for xb, yb in loader:
        logits, aux = model(_with_feature_dim(xb).to(device))
        preds.append(logits.argmax(dim=1).cpu())
        targets.append(yb)
        if is_spiking:
            spike_chunks.append(aux.detach().cpu())

    metrics = classification_metrics(torch.cat(targets).numpy(),
                                     torch.cat(preds).numpy(),
                                     num_classes=num_classes)

    if is_spiking:
        counts = snn_operation_counts(torch.cat(spike_chunks), recurrent_fanout(model),
                                      k, num_classes)
    else:
        counts = lstm_operation_counts(k, model.hidden_size, num_classes)

    metrics.update(counts)
    metrics["k"] = k
    metrics["latency_fraction"] = k / dataset.seq_len
    metrics["energy_pj"] = energy_pj(counts["macs"], counts["sops"])
    return metrics
