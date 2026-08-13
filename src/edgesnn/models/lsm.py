"""Liquid State Machine baseline: the same spiking substrate, readout-only training.

The LSM is the control that makes the second research question answerable. It is
a `TruncatedSpikingReservoir` in every respect -- identical LIF dynamics,
identical `N_res`, identical connectivity `p_c`, identical initializer, identical
truncation window `k` -- except that `W_in` and `W_res` are held at their initial
values and only the linear readout `W_out` is trained, in the classical reservoir
computing manner.

Because it is implemented as a subclass rather than a reimplementation, that
identity holds by construction: there is exactly one copy of the dynamics in this
codebase, so the two models cannot silently drift apart and the difference
between them is attributable to reservoir training alone, with no confounding
change in architecture, neuron model or connectivity.

Three consequences worth stating explicitly, because each is a reportable result
rather than an implementation detail:

- **Trainable capacity collapses.** Only `C * N_res + C` parameters remain: 514
  at `C=2` and 2,570 at `C=10`, two orders of magnitude below the LSTM and the
  proposed model. The LSM is therefore not a fair accuracy competitor and is not
  meant to be one; it measures what the random projection alone achieves.
- **The surrogate gradient plays no part in fitting it.** With the reservoir
  frozen, no gradient is ever propagated through a spike, so the LSM's accuracy
  is independent of the surrogate's shape or its `beta`. It is consequently also
  a clean control against the claim that results are an artefact of the
  particular surrogate chosen.
- **Its spike statistics are a pure function of the initializer.** Since the
  reservoir never changes, the SOP count of a trained LSM is a property of the
  initialization method alone -- which makes it the natural place to read off
  how much energy each cheap generator implies before learning is allowed to
  perturb the firing regime.
"""

import torch

from .spiking_reservoir import TruncatedSpikingReservoir


class LiquidStateMachine(TruncatedSpikingReservoir):
    """Frozen spiking reservoir with a trained linear readout."""

    def __init__(self, input_size, hidden_size, num_classes, truncation_k=40, **neuron_kwargs):
        super().__init__(input_size, hidden_size, num_classes,
                         truncation_k=truncation_k, **neuron_kwargs)
        self.freeze_reservoir()

    def freeze_reservoir(self):
        """Detach `W_in` and `W_res` from the optimizer.

        Called again after `load_custom_initialization` would be redundant --
        copying values into a tensor does not alter its `requires_grad` -- but the
        method is public so a training script can assert on it rather than trust
        it.
        """
        for param in self.lif_layer.parameters():
            param.requires_grad_(False)

    def reservoir_is_frozen(self):
        """True when no reservoir weight would receive a gradient."""
        return not any(p.requires_grad for p in self.lif_layer.parameters())

    def forward(self, x):
        """Identical to the parent's, with the reservoir run outside autograd.

        Skipping the graph is not merely an optimization: with `W_in` and
        `W_res` frozen there is nothing along the recurrent path for BPTT to
        update, so building the `k`-step graph would allocate memory
        proportional to the truncation window purely to discard it. The readout
        still receives its gradient, since `W_out` is applied outside the
        `no_grad` block and the accumulated spike count enters it as a fixed
        feature vector. Numerically the spike counts are the same tensors the
        parent would produce.
        """
        with torch.no_grad():
            spike_count = self.reservoir_forward(x)

        logits = self.W_out(spike_count)
        return logits, spike_count
