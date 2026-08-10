import torch
import torch.nn as nn


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

        self.surrogate_spike = FastSigmoidSurrogate.apply

    def forward(self, x_t, V_prev):
        """
        Calculates the discrete temporal dynamics for a single time step.
        (Referenced in Section 3.1: Discrete-Time LIF Formulation)
        """
        # 1. Integrate incoming currents (Input + Recurrent Spikes)
        I_t = self.W_in(x_t) + self.W_res(V_prev)

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

    def load_custom_initialization(self, w_in_tensor, w_res_tensor):
        """
        Injects the cheap PRNG/LDS weights into the reservoir, overriding PyTorch defaults.
        """
        with torch.no_grad():
            self.lif_layer.W_in.weight.copy_(w_in_tensor)
            self.lif_layer.W_res.weight.copy_(w_res_tensor)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, features)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize membrane potential (V_0)
        V_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        spike_count = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # TRUNCATED LEARNING: We only run the loop up to k, ignoring the rest of seq_len
        # This simulates hardware powering down early to save energy.
        actual_steps = min(self.k, seq_len)

        for t in range(actual_steps):
            x_t = x[:, t, :]
            V_t, S_t = self.lif_layer(x_t, V_t)

            # Accumulate spikes for the readout layer (Rate Coding)
            spike_count += S_t

        # The final prediction is a linear mapping of the total spike counts
        logits = self.W_out(spike_count)

        return logits, spike_count