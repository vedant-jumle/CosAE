from transformers import PretrainedConfig, PreTrainedModel


class CosAEConfig(PretrainedConfig):
    model_type = "cosae"

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        # Encoder parameters
        in_channels: int = 3,
        hidden_dims: list[int] = (64, 128, 256, 512),
        num_res_blocks: int = 2,
        downsample_strides: list[int] = (2, 2, 2, 2),
        use_encoder_attention: bool = True,
        encoder_attention_heads: int = 8,
        encoder_attention_layers: int = 1,
        bottleneck_channels: int = 256,
        basis_size: int = 32,
        norm_type: str = "gn",      # "gn" (GroupNorm) or "ln" (LayerNorm)
        activation: str = "gelu",   # "gelu" or "silu"

        # Decoder parameters
        decoder_hidden_dim: int = 256,
        decoder_upsample_strides: list[int] = (2,),  # e.g. (2,) for one 2× upsample
        use_decoder_attention: bool = False,
        decoder_attention_heads: int = 8,
        decoder_attention_layers: int = 0,

        **kwargs,
    ):
        """
        Configuration for CosAEModel, including encoder, HCM, and decoder settings.
        """
        super().__init__(**kwargs)

        # Encoder settings
        self.in_channels = in_channels
        self.hidden_dims = list(hidden_dims)
        self.num_res_blocks = num_res_blocks
        self.downsample_strides = list(downsample_strides)
        self.use_encoder_attention = use_encoder_attention
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_attention_layers = encoder_attention_layers
        self.bottleneck_channels = bottleneck_channels
        self.basis_size = basis_size
        self.norm_type = norm_type
        self.activation = activation
        self.image_size = image_size
        
        # Decoder settings
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_upsample_strides = list(decoder_upsample_strides)
        self.use_decoder_attention = use_decoder_attention
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_attention_layers = decoder_attention_layers
        