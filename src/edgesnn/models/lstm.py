"""Dense LSTM baseline.

This is the accuracy target the spiking models must match. It is deliberately a
plain single-layer LSTM with a linear readout: the thesis argues that a cheaply
initialized, truncated SNN can rival a conventional recurrent network at a
fraction of the energy, and that argument is only interesting if the
conventional network is the ordinary one a practitioner would reach for.

Two design choices make the comparison fair rather than merely convenient.

1. **Matched parameter budget.** An LSTM spends four times as many parameters
   per hidden unit as a reservoir does, because each unit carries input, forget,
   cell and output gates. A hidden size of H therefore costs roughly 4H^2
   recurrent parameters against the reservoir's N_res^2, so parity is reached at
   H ~= N_res / 2. With N_res = 256 and H = 128 both models sit near 66-67k
   trainable parameters, and neither can be accused of having simply been given
   more capacity.

2. **The same truncation window.** `truncation_k` cuts the LSTM's input
   sequence exactly as it cuts the SNN's. Sweeping k for both models is what
   separates "the data permits early classification" from "the SNN handles
   truncation well" — without the LSTM curve, a flat SNN accuracy-versus-k plot
   could be explained entirely by the data and not at all by the method.

The energy side of the comparison is analytic rather than measured: an LSTM
performs the same dense multiply-accumulates on every input regardless of its
values, so `mac_count()` is exact and data-independent. This is precisely the
asymmetry the thesis exploits, since the SNN's cost depends on how many spikes
actually occur.
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Single-layer LSTM followed by a linear readout.

    Args:
        input_size: features per time step (1 for the univariate UCR series).
        hidden_size: LSTM hidden units H. Defaults to 128, which pairs with
            N_res = 256 at a matched parameter budget.
        num_classes: number of output logits.
        truncation_k: process only the first k time steps. `None` uses the whole
            sequence. Shares the semantics of
            `TruncatedSpikingReservoir.truncation_k`, so a k-sweep means the same
            thing for both models.
    """

    def __init__(self, input_size=1, hidden_size=128, num_classes=2,
                 truncation_k=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.k = truncation_k

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                            batch_first=True)
        self.W_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """x: (batch, seq_len, features) -> (logits, steps_processed).

        The second return value mirrors the spiking model's `(logits,
        spike_count)` signature so that a shared training loop can treat both
        models identically; here it is the number of time steps actually
        consumed, which is what the analytic energy model needs.
        """
        _, seq_len, _ = x.size()
        steps = seq_len if self.k is None else min(self.k, seq_len)

        # The classification reads the final hidden state of the truncated
        # window. For the SNN the analogous quantity is the accumulated spike
        # count over the same window; both summarise the window in one vector.
        out, _ = self.lstm(x[:, :steps, :])
        logits = self.W_out(out[:, -1, :])

        return logits, steps

    def mac_count(self, seq_len=None):
        """Multiply-accumulates for one forward pass over the truncated window.

        An LSTM cell computes four gate pre-activations, each of which applies
        both an input matrix (H x input_size) and a recurrent matrix (H x H).
        Every one of these is dense and executes for every input, so the count
        is exact rather than an estimate.
        """
        steps = seq_len if self.k is None else min(self.k, seq_len or self.k)
        per_step = 4 * self.hidden_size * (self.input_size + self.hidden_size)
        return {
            "lstm": per_step * steps,
            "readout": self.hidden_size * self.num_classes,
            "total": per_step * steps + self.hidden_size * self.num_classes,
        }

    def trainable_parameters(self):
        """Trainable parameter count, for the matched-budget table."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
