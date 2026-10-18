"""
End-to-end encoding pipeline: Image -> Feature Maps -> StimulusProtocol.

Every call to encode_to_protocol() returns a validated StimulusProtocol.
This is the ONLY public encoding entry point for downstream consumers.
"""

import numpy as np
import torch
from .cnn_feature_extractor import CNNFeatureExtractor
from .delayed_phase_encoder import DelayedPhaseEncoder
from .lif_encoder import LIFEncoder
from .rate_encoder import RateEncoder
from .stimulus_schema import (
    StimulusProtocol, trim_to_charge_budget, DEFAULT_TRIM_TARGET_NC,
)


class EncodingPipeline:
    """
    Image -> CNN features -> StimulusProtocol.

    Usage:
        pipeline = EncodingPipeline(cnn_model, encoder_type="delayed_phase")
        protocol = pipeline.encode_to_protocol(image_tensor)
        protocol.validate()  # always valid; raises if not
    """

    VALID_ENCODERS = ("delayed_phase", "lif", "rate")

    def __init__(
        self,
        cnn_model,
        encoder_type="delayed_phase",
        encoder_params=None,
        device="cpu",
        trim_target_nc=DEFAULT_TRIM_TARGET_NC,
        trim_warn=True,
    ):
        """
        Args:
            cnn_model: feature extractor.
            encoder_type: one of VALID_ENCODERS.
            encoder_params: dict forwarded to the encoder ctor.
            device: torch device for CNN inference.
            trim_target_nc: cumulative biphasic charge budget per protocol.
                Default ~180 nC (90% of SAFETY_MAX_TOTAL_CHARGE_PER_TRIAL_NC).
                Set to None to skip trimming (for research only — protocol
                may then fail validate_safety downstream).
            trim_warn: if True, print a one-line notice the first time a
                pulse is actually dropped so the user knows trimming is
                active. Subsequent trims are silent.
        """
        if encoder_type not in self.VALID_ENCODERS:
            raise ValueError(
                f"encoder_type must be one of {self.VALID_ENCODERS}, "
                f"got '{encoder_type}'"
            )

        self.cnn = cnn_model
        self.cnn.eval()
        self.device = device
        self.encoder_type = encoder_type
        self.trim_target_nc = trim_target_nc
        self.trim_warn = trim_warn
        self._trim_warned = False

        params = encoder_params or {}
        if encoder_type == "delayed_phase":
            self.encoder = DelayedPhaseEncoder(**params)
        elif encoder_type == "lif":
            self.encoder = LIFEncoder(**params)
        elif encoder_type == "rate":
            self.encoder = RateEncoder(**params)

    @torch.no_grad()
    def encode_to_protocol(self, image, duration_ms=2000.0):
        """
        Encode a single image into a StimulusProtocol.

        Args:
            image: Tensor or ndarray, shape (H, W) or (1, H, W) or (1, 1, H, W).
                   Values in [0, 1].
            duration_ms: Encoding duration.

        Returns:
            StimulusProtocol (validated).
        """
        feature_maps = self._extract_features(image)

        if self.encoder_type == "delayed_phase":
            proto = self.encoder.encode_to_protocol(feature_maps)
        elif self.encoder_type == "lif":
            output = self.encoder.encode_features(feature_maps, duration_ms)
            proto = LIFEncoder.spike_output_to_protocol(output)
        elif self.encoder_type == "rate":
            output = self.encoder.encode_features(feature_maps, duration_ms)
            proto = RateEncoder.rate_output_to_protocol(output)

        if self.trim_target_nc is not None:
            n_before = proto.total_pulses
            proto = trim_to_charge_budget(proto, self.trim_target_nc)
            n_after = proto.total_pulses
            if n_after < n_before and self.trim_warn and not self._trim_warned:
                print(f"[EncodingPipeline] trimmed {n_before - n_after} pulse(s) "
                      f"({n_before}->{n_after}) to fit {self.trim_target_nc:.0f} nC budget "
                      f"({self.encoder_type}). Further trims will be silent.")
                self._trim_warned = True

        return proto

    @torch.no_grad()
    def encode_batch_to_protocol(self, images, duration_ms=2000.0):
        """
        Encode a batch of images into StimulusProtocol list.

        Args:
            images: (batch, 1, H, W) tensor.

        Returns:
            List of StimulusProtocol.
        """
        return [
            self.encode_to_protocol(images[i], duration_ms)
            for i in range(images.shape[0])
        ]

    @torch.no_grad()
    def _extract_features(self, image):
        """Image tensor -> (n_channels, h, w) numpy feature maps."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        features = self.cnn.extract_features(image)
        return features[0].cpu().numpy()
