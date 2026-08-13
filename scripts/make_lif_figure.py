"""Regenerate Figure 2.2 of the thesis: discrete-time LIF neuron dynamics.

The figure is produced by running the *actual* `LIFCell` used in every
experiment, not by redrawing the equations by hand. That is the whole point:
the previous version of this figure annotated a reset-to-zero rule
(`V[t] = V_reset`) while the implementation used reset-by-subtraction, and a
hand-drawn figure is exactly the kind of artefact that can drift away from the
code without anyone noticing. Generating it from `LIFCell` makes that class of
error impossible.

A single neuron is used with no recurrent connection, so the trace shows the
input-driven dynamics alone. Run from the repository root:

    uv run python scripts/make_lif_figure.py
"""

import torch

from edgesnn.models.spiking_reservoir import LIFCell
from edgesnn.plotting import plot_lif_dynamics

# Written into the Overleaf-synced thesis folder, replacing the old figure.
FIGURE_PATH = (
    "reports/latex/Optimizing_Initialization_and_Temporal_Learning_in_Edge_"
    "Spiking_Neural_Networks/images/LIF_Dynamics.png"
)

THRESHOLD = 1.0
LEAK_RATE = 0.9

# A deliberately shaped input. Each burst builds the potential up subthreshold
# so the leak is visible between inputs, then drives it well past the threshold
# so that the charge retained by the subtraction is large enough to read off the
# plot: a marginal crossing would leave a residual near zero and look
# indistinguishable from a hard reset, which is the very confusion this figure
# exists to prevent. Silence afterwards shows the decay back towards rest.
CURRENT = [
    0.0, 0.25, 0.25, 0.25, 0.0, 0.90, 0.0, 0.0, 0.0, 0.0,
    0.40, 0.40, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0,
]


def main():
    torch.manual_seed(0)

    cell = LIFCell(input_size=1, hidden_size=1, leak_rate=LEAK_RATE,
                   threshold=THRESHOLD, reset_potential=0.0)
    with torch.no_grad():
        # Unit input weight and no recurrence: the trace then reflects the
        # neuron model itself rather than a particular random initialization.
        cell.W_in.weight.fill_(1.0)
        cell.W_res.weight.fill_(0.0)

    V = torch.zeros(1, 1)
    S = torch.zeros(1, 1)
    potentials, spikes = [], []

    with torch.no_grad():
        for value in CURRENT:
            x_t = torch.tensor([[value]])
            V, S = cell(x_t, V, S)
            potentials.append(V.item())
            spikes.append(S.item())

    print(f"spikes at steps: {[t for t, s in enumerate(spikes) if s > 0]}")
    print(f"peak potential:  {max(potentials):.3f}")

    plot_lif_dynamics(potentials, spikes, CURRENT, threshold=THRESHOLD,
                      reset_potential=0.0, leak_rate=LEAK_RATE,
                      save_path=FIGURE_PATH, show=False)


if __name__ == "__main__":
    main()
