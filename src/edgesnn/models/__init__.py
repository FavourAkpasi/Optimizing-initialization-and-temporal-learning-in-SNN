"""Model definitions for the Edge SNN experiments.

    spiking_reservoir  Truncated spiking recurrent reservoir (LIF + surrogate grad).
    lstm               Dense LSTM baseline (parameter-matched to the reservoir).
    lsm                Liquid State Machine baseline (same substrate, readout only).
"""

from edgesnn.models.lsm import LiquidStateMachine
from edgesnn.models.lstm import LSTMBaseline
from edgesnn.models.spiking_reservoir import (
    FastSigmoidSurrogate,
    LIFCell,
    TruncatedSpikingReservoir,
)

__all__ = [
    "FastSigmoidSurrogate",
    "LIFCell",
    "TruncatedSpikingReservoir",
    "LSTMBaseline",
    "LiquidStateMachine",
]
