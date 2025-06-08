# CosAE: Convolutional Harmonic Autoencoder
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-cosae-blue?logo=Huggingface)](https://huggingface.co/vedant-jumle/cosae) [![Paper](https://img.shields.io/badge/Paper-Sifei%20et%20al.%202024-blue?logo=google-scholar)](https://research.nvidia.com/labs/amri/publication/sifei2024cosae/)

CosAE is a PyTorch implementation of the Convolutional Harmonic Autoencoder (CosAE). It encodes images into learnable harmonic representations (amplitudes and phases), constructs spatial cosine bases via a Harmonic Construction Module, and decodes back to RGB images. This repository provides the core model code, a Jupyter notebook for training and evaluation, and pretrained weights in SafeTensors format.

## Features
- Convolutional encoder with residual blocks and optional global attention
- Harmonic Construction Module (HCM) learning per-channel frequencies
- Decoder with upsampling and optional global attention for reconstruction
- Supports raw RGB input or augmented FFT channels (RGB + FFT)
- Pretrained model weights available (SafeTensors)
- Training and tracking via Jupyter notebook and Weights & Biases

## Installation
### Requirements
```bash
pip install -r requirements.txt
```

## Quickstart
### Inference with Pretrained Model
```python
import torch
from cosae import CosAEModel

# Load pretrained CosAE (expects 9-channel input: 3 RGB + 6 FFT)
model = CosAEModel.from_pretrained("model/model-final")
model.eval()

# Dummy input: batch of 1, 9 channels, 256×256
x = torch.randn(1, 9, 256, 256)
with torch.no_grad():
    recon = model(x)  # recon.shape == [1, 3, 256, 256]
```

### Preparing Raw Images
```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
img = Image.open("path/to/image.png").convert("RGB")
tensor = transform(img).unsqueeze(0)  # [1, 3, 256, 256]
```

### Generating FFT Channels (Optional)
To use FFT-augmented input (in_channels=9), compute the 2D FFT per RGB channel and stack real/imaginary parts:
```python
import torch
fft = torch.fft.rfft2(tensor, norm="ortho")        # complex tensor [1,3,H,W/2+1]
real, imag = fft.real, fft.imag
# Optionally pad or reshape to match full spatial dims
x9 = torch.cat([tensor, real, imag], dim=1)         # [1,9,H,W]
```

## Creating a Model from Scratch
Use the `CosAEConfig` and `CosAEModel` classes to instantiate a model with custom settings:
```python
from cosae.config import CosAEConfig
from cosae.cosae import CosAEModel

# 1) Define a configuration (example uses RGB input only)
config = CosAEConfig(
    in_channels=3,
    hidden_dims=[64, 128, 256, 512],
    downsample_strides=[2, 2, 2, 2],
    use_encoder_attention=True,
    encoder_attention_heads=8,
    encoder_attention_layers=1,
    bottleneck_channels=256,
    basis_size=16,
    decoder_hidden_dim=256,
    decoder_upsample_strides=[2],
    use_decoder_attention=False,
)

# 2) Instantiate the model
model = CosAEModel(config)

# 3) (Optional) Save the model and config for later reuse
model.save_pretrained("./my_cosae_model")  # creates model-final.safetensors and config.json
```

## Training and Evaluation
A full training and evaluation pipeline is provided in `cosine-ae.ipynb`. Launch Jupyter to run experiments and track metrics with Weights & Biases:
```bash
jupyter lab cosine-ae.ipynb
```

## Repository Structure
```
.
├── cosae/               # Core model implementation (config, encoder, HCM, decoder)
├── model/               # Pretrained weights and config (SafeTensors)
├── cosine-ae.ipynb      # Notebook: training, evaluation, and demos
├── LICENSE              # MIT License
└── README.md            # Project overview and usage
```

## References
- Sifei et al. (2024). CosAE: Convolutional Harmonic Autoencoder. NVIDIA AMRI. https://research.nvidia.com/labs/amri/publication/sifei2024cosae/

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to fork the repository and submit pull requests.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
