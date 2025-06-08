import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import PreTrainedModel
from .modules import *
from .config import CosAEConfig

class CosAEModel(PreTrainedModel):
    config_class = CosAEConfig
    base_model_prefix = "cosae"

    def __init__(self, config: CosAEConfig):
        super().__init__(config)
        # 1) Encoder
        self.encoder = CosAEEncoder(config)

        # 2) Harmonic Construction Module
        #    derive P = total downsampling factor from encoder strides
        stem_ds = 2 * 2
        P       = stem_ds * math.prod(config.downsample_strides)
        #    basis size T = P // 2
        T = P // 2
        self.T = T
        self.hcm = HarmonicConstructionModule(
            bottleneck_channels=config.bottleneck_channels,
            basis_size=config.basis_size
        )

        # 3) Decoder
        self.decoder = CosAEDecoder(config)

        # initialize weights, etc.
        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
          pixel_values: [B, C_in, H, W]  (C_in = 3 or 9 if using FFT)
        Returns:
          recon:        [B, 3, H, W]      reconstructed image
        """
        # Encode to get amplitudes & phases
        bottleneck = self.encoder(pixel_values)          # [B, 2c, H', W']
        amp, ph = torch.chunk(bottleneck, 2, dim=1)      # each [B, c, H', W']

        # Build harmonics
        harmonics = self.hcm(amp, ph)                    # [B, c, H, W]

        # Decode to reconstruct
        recon = self.decoder(harmonics)                  # [B, 3, H, W]
        return recon