"""
Encoder modules for converting images to stimulation protocols.

All encoders ultimately produce a StimulusProtocol — the frozen unified
format consumed by Intan TCP, integration tests, and real organoid experiments.
"""

from .stimulus_schema import (
    Pulse, StimulusProtocol, euclidean_distance, trim_to_charge_budget,
)
from .cnn_feature_extractor import CNNFeatureExtractor
from .delayed_phase_encoder import DelayedPhaseEncoder
from .lif_encoder import LIFEncoder
from .rate_encoder import RateEncoder
from .pipeline import EncodingPipeline
