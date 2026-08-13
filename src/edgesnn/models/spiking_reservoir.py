import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Surrogate gradients
#
# Every one of these shares the SAME forward pass -- a hard Heaviside spike, so
# the network's inference behaviour is bit-identical across families and the
# choice affects learning only. They differ solely in the derivative substituted
# for the step function's (zero almost everywhere, undefined at threshold)
# during the backward pass.
#
# NOTE ON SCALE: at x = 0 the four families return 1, beta/2, 1 and beta/4
# respectively, so at the default beta = 10 their gradient magnitudes differ by
# up to 5x. The surrogate family therefore CANNOT be tuned independently of the
# learning rate -- the two must be crossed in the grid, or a family will be
# rejected for having the wrong gradient scale rather than the wrong shape.
# ---------------------------------------------------------------------------


class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast-sigmoid surrogate: `1 / (1 + beta|x|)^2`. The project default."""

    @staticmethod
    def forward(ctx, membrane_potential, threshold, beta):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.beta = beta

        # Heaviside Step Function: Emit 1.0 if potential >= threshold, else 0.0
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        x = membrane_potential - ctx.threshold
        grad = 1.0 / (1.0 + ctx.beta * torch.abs(x)) ** 2
        return grad_output * grad, None, None


class ArcTanSurrogate(torch.autograd.Function):
    """Arctangent surrogate: `beta / (2 (1 + (pi beta x / 2)^2))`.

    Heavier tails than the fast sigmoid, so neurons far from threshold keep
    receiving a gradient instead of being effectively frozen out.
    """

    @staticmethod
    def forward(ctx, membrane_potential, threshold, beta):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        x = membrane_potential - ctx.threshold
        grad = ctx.beta / (2.0 * (1.0 + (math.pi / 2.0 * ctx.beta * x) ** 2))
        return grad_output * grad, None, None


class TriangularSurrogate(torch.autograd.Function):
    """Piecewise-linear surrogate: `max(0, 1 - beta|x|)`.

    The cheapest of the four on hardware -- a comparison and a subtraction, no
    division or transcendental -- and the only one with compact support, so
    neurons outside the window receive exactly zero gradient. That makes it the
    most interesting family for this thesis: if it matches the others, the
    backward pass is as amenable to cheap arithmetic as the forward pass.
    """

    @staticmethod
    def forward(ctx, membrane_potential, threshold, beta):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        x = membrane_potential - ctx.threshold
        grad = torch.clamp(1.0 - ctx.beta * torch.abs(x), min=0.0)
        return grad_output * grad, None, None


class SigmoidSurrogate(torch.autograd.Function):
    """Logistic-derivative surrogate: `beta * s(1-s)` with `s = sigmoid(beta x)`."""

    @staticmethod
    def forward(ctx, membrane_potential, threshold, beta):
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        s = torch.sigmoid(ctx.beta * (membrane_potential - ctx.threshold))
        grad = ctx.beta * s * (1.0 - s)
        return grad_output * grad, None, None


SURROGATES = {
    "fast_sigmoid": FastSigmoidSurrogate,
    "arctan": ArcTanSurrogate,
    "triangular": TriangularSurrogate,
    "sigmoid": SigmoidSurrogate,
}


def get_surrogate(name):
    """Look up a surrogate family by name, failing loudly on a typo."""
    try:
        return SURROGATES[name]
    except KeyError:
        raise ValueError(
            f"Unknown surrogate {name!r}; expected one of {sorted(SURROGATES)}."
        ) from None


class LIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, leak_rate=0.9, threshold=1.0,
                 reset_potential=0.0, surrogate="fast_sigmoid", surrogate_beta=10.0):
        super(LIFCell, self).__init__()
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.surrogate_name = surrogate
        self.surrogate_beta = surrogate_beta

        # Trainable Recurrent Weights (Will be initialized via custom methods)
        self.W_res = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)

        # Structural sparsity mask for W_res. Registered as a buffer (not a
        # parameter) so it is saved with the model, moves with .to(device), and
        # is never updated by the optimizer. `None` means a dense reservoir.
        self.register_buffer("res_mask", None)

        self.surrogate_spike = get_surrogate(surrogate).apply

    def set_sparsity_mask(self, mask):
        """Fix which recurrent synapses exist, for the lifetime of the model."""
        if mask is not None:
            mask = mask.to(dtype=self.W_res.weight.dtype,
                           device=self.W_res.weight.device)
        self.res_mask = mask

    def recurrent_weight(self):
        """W_res with absent synapses forced to zero.

        Masking inside the forward pass (rather than only at initialization)
        keeps the connectivity structure fixed during training: the gradient
        w.r.t. a masked entry is multiplied by the same zero, so a synapse that
        does not exist can never be learned into existence by BPTT. This is what
        makes the reported SOP count valid for the trained model and not merely
        for its initialization.
        """
        if self.res_mask is None:
            return self.W_res.weight
        return self.W_res.weight * self.res_mask

    def forward(self, x_t, V_prev, S_prev):
        """
        Calculates the discrete temporal dynamics for a single time step.
        (Referenced in Section 2.2.1: Discrete-Time LIF Formulation)

        The recurrent term consumes the previous BINARY spike vector S_prev, not
        the membrane potential. This is the defining property of a spiking
        recurrent layer: neurons communicate through 1-bit events, so a
        recurrent synapse costs an accumulate (a Synaptic Operation) rather than
        a multiply-accumulate. Feeding V_prev back instead would make the layer
        a conventional dense RNN and invalidate the SOP-based energy analysis.
        """
        # 1. Integrate incoming currents (Input + Recurrent Spikes)
        I_t = self.W_in(x_t) + F.linear(S_prev, self.recurrent_weight())

        # 2. Leaky integration
        H_t = (V_prev - self.reset_potential) * self.leak_rate + I_t

        # 3. Fire: Check if potential exceeds threshold
        S_t = self.surrogate_spike(H_t, self.threshold, self.surrogate_beta)

        # 4. Reset: Reset-by-subtraction mechanism
        # If a spike occurs (S_t=1), subtract the threshold from the potential
        V_t = H_t - (S_t * self.threshold)

        return V_t, S_t


class TruncatedSpikingReservoir(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, truncation_k=40,
                 leak_rate=0.9, threshold=1.0, reset_potential=0.0,
                 surrogate="fast_sigmoid", surrogate_beta=10.0):
        """
        A full Spiking Recurrent Network with an explicit truncation window k.

        The neuron and surrogate hyperparameters are constructor arguments rather
        than constants so the tuning grid can reach them. They are nuisance
        parameters: selected once per model on the validation split at k = T, then
        frozen across the truncation and initialization sweeps. Re-tuning them at
        each k would mix tuning effects into the accuracy-versus-k curve.
        """
        super(TruncatedSpikingReservoir, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.k = truncation_k

        # The Spiking Reservoir
        self.lif_layer = LIFCell(input_size, hidden_size, leak_rate=leak_rate,
                                 threshold=threshold, reset_potential=reset_potential,
                                 surrogate=surrogate, surrogate_beta=surrogate_beta)

        # The Trainable Readout Layer
        self.W_out = nn.Linear(hidden_size, num_classes)

    def load_custom_initialization(self, w_in_tensor, w_res_tensor, res_mask=None):
        """
        Injects the cheap PRNG/LDS weights into the reservoir, overriding PyTorch defaults.

        `res_mask` records which recurrent synapses exist. Pass the same mask
        that `initialize_reservoir(..., connectivity=p_c)` applied, so the zeros
        it introduced are held at zero for the whole of training instead of
        being filled in by the first gradient step.
        """
        with torch.no_grad():
            self.lif_layer.W_in.weight.copy_(w_in_tensor)
            self.lif_layer.W_res.weight.copy_(w_res_tensor)
        if res_mask is not None:
            self.lif_layer.set_sparsity_mask(res_mask)

    def synapse_count(self):
        """Existing synapses per matrix — the fan-out a spike pays for in SOPs."""
        mask = self.lif_layer.res_mask
        recurrent = (int(mask.sum().item()) if mask is not None
                     else self.lif_layer.W_res.weight.numel())
        return {
            "W_in": self.lif_layer.W_in.weight.numel(),
            "W_res": recurrent,
            "W_out": self.W_out.weight.numel(),
        }

    def reservoir_forward(self, x):
        """Run the reservoir over the truncated window; return the spike count.

        Split out of `forward` so a readout-only model (`models/lsm.py`) can
        reuse this loop verbatim rather than reimplementing the dynamics. That
        the LSM and the proposed model share one substrate is then true by
        construction, which is what licenses attributing the difference between
        them solely to whether the reservoir is trained.

        x shape: (batch_size, sequence_length, features)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize membrane potential (V_0) and the spike vector fed back at t=0.
        V_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        S_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        spike_count = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # TRUNCATED LEARNING: We only run the loop up to k, ignoring the rest of seq_len
        # This simulates hardware powering down early to save energy.
        actual_steps = min(self.k, seq_len)

        for t in range(actual_steps):
            x_t = x[:, t, :]
            V_t, S_t = self.lif_layer(x_t, V_t, S_t)

            # Accumulate spikes for the readout layer (Rate Coding)
            spike_count += S_t

        return spike_count

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, features)
        """
        spike_count = self.reservoir_forward(x)

        # The final prediction is a linear mapping of the total spike counts
        logits = self.W_out(spike_count)

        return logits, spike_count

    def trainable_parameters(self):
        """Parameter count, for the capacity comparison against the baselines."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)