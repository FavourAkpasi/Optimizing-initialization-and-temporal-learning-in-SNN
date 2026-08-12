import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Custom autograd function to solve the non-differentiable Heaviside step function.
    Forward pass: Hard step function (Spike emission).
    Backward pass: Derivative of a fast sigmoid (Surrogate gradient).
    """

    @staticmethod
    def forward(ctx, membrane_potential, threshold):
        # Save potential for the backward pass
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold

        # Heaviside Step Function: Emit 1.0 if potential >= threshold, else 0.0
        spike = (membrane_potential >= threshold).float()
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold

        # Fast Sigmoid derivative parameters (beta determines steepness)
        beta = 10.0

        # Surrogate gradient calculation: f'(x) = 1 / (1 + |beta * x|)^2
        grad = 1.0 / (1.0 + beta * torch.abs(membrane_potential - threshold)) ** 2

        # Chain rule
        return grad_output * grad, None


class LIFCell(nn.Module):
    def __init__(self, input_size, hidden_size, leak_rate=0.9, threshold=1.0, reset_potential=0.0):
        super(LIFCell, self).__init__()
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.threshold = threshold
        self.reset_potential = reset_potential

        # Trainable Recurrent Weights (Will be initialized via custom methods)
        self.W_res = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)

        # Structural sparsity mask for W_res. Registered as a buffer (not a
        # parameter) so it is saved with the model, moves with .to(device), and
        # is never updated by the optimizer. `None` means a dense reservoir.
        self.register_buffer("res_mask", None)

        self.surrogate_spike = FastSigmoidSurrogate.apply

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
        S_t = self.surrogate_spike(H_t, self.threshold)

        # 4. Reset: Reset-by-subtraction mechanism
        # If a spike occurs (S_t=1), subtract the threshold from the potential
        V_t = H_t - (S_t * self.threshold)

        return V_t, S_t


class TruncatedSpikingReservoir(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, truncation_k=40):
        """
        A full Spiking Recurrent Network with an explicit truncation window k.
        """
        super(TruncatedSpikingReservoir, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.k = truncation_k

        # The Spiking Reservoir
        self.lif_layer = LIFCell(input_size, hidden_size)

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