"""
Feature extraction from neural recordings for decoder input.

Methods:
1. Time-bin firing count (Wang et al. 2025)
2. RMS / Power (Donati et al. 2026)
3. Spike rate
4. PCA dimensionality reduction
"""

from .extractors import (
    extract_timebin_features,
    extract_rms_features,
    extract_power_features,
    extract_spike_rate_features,
    reduce_with_pca,
)
