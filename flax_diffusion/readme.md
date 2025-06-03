# Installation
The following instructions will guide you to run StableDiffusion1.4 using Jax/Flax on a NVIDIA CUDA supported GPU.
## Requirements
- Python 3.11
- CUDA supported GPU with at least 6GB vRAM
## On WSL + CUDA GPU
1. Install required packages by `pip install -r requirements.txt`
2. Install Cuda Toolkit for WSL Ubuntu [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network).

## Getting started
1. Make sure the following code works by running `python flax_test_sd.py`, this will download StableDiffusion1.4 and test one inference sample using GPU.