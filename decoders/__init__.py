"""
Decoder modules for classifying neural responses.

Implements:
1. SNN decoder (Donati et al. 2026) - single-layer LIF with synaptic conductance
2. Logistic Regression (Wang et al. 2025) - PCA + multinomial LR
3. SVM baseline (Donati et al. 2026) - linear kernel
4. MLP baseline (Donati et al. 2026) - single hidden layer
"""

from .snn_decoder import SNNDecoder
from .conventional_decoders import LogisticRegressionDecoder, SVMDecoder, MLPDecoder
