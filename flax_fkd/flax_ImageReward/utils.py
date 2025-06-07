import os
import numpy as np
from typing import Tuple, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
import ImageReward as RM
from PIL import Image
from flax_BLIP import FlaxBLIP_Pretrain

class FlaxImageRewardPreprocessor(nn.Module):
    n_px: int = 224
    dtype: jnp.dtype = jnp.float32
    mean: Tuple[float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float] = (0.26862954, 0.26130258, 0.27577711)

    def __call__(
        self,
        image_input: Union[Image.Image, jnp.ndarray, str]
    ):
        """
        Calls Flax image preprocessing transformation pipeline
        Input: PIL Image or HWC jnp.ndarray (self.dtype, [0,1] or uint8 [0,255]), or str (path to file).
        Output: CHW jnp.ndarray, normalized and ready for model input.
        """
        if isinstance(image_input, Image.Image):
            img_pil = image_input.convert("RGB")
            img_hwc_f32 = np.array(img_pil).astype(self.dtype) / 255.0
        elif isinstance(image_input, jnp.ndarray):
            if image_input.dtype == jnp.uint8:
                img_hwc_f32 = image_input.astype(self.dtype) / 255.0
            else: # Assume float32, float64 or bfloat16, [0,1]
                img_hwc_f32 = image_input
            if img_hwc_f32.ndim == 2: # Grayscale
                img_hwc_f32 = jnp.stack([img_hwc_f32]*3, axis=-1)
            elif img_hwc_f32.shape[-1] == 1: # Grayscale (H,W,1)
                img_hwc_f32 = jnp.repeat(img_hwc_f32, 3, axis=-1)
            elif img_hwc_f32.shape[-1] == 4: # RGBA
                img_hwc_f32 = img_hwc_f32[..., :3] # Take RGB
        elif isinstance(image_input, str):
            img_pil = Image.open(image_input)
            img_hwc_f32 = np.array(img_pil).astype(self.dtype) / 255.0
        else:
            raise TypeError("Input image must be PIL.Image or JAX HWC ndarray.")

        img_resized = self._jax_resize_pil_like(img_hwc_f32, self.n_px)
        img_cropped = self._jax_center_crop(img_resized, self.n_px)
        img_chw = self._jax_to_chw_tensor(img_cropped)
        img_normalized = self._jax_normalize(img_chw, self.mean, self.std)
        return img_normalized

    def _jax_resize_pil_like(self, image_hwc_f32: jnp.ndarray, shorter_edge_size: int) -> jnp.ndarray:
        """Resizes an image (HWC, self.dtype) so the shorter edge is `shorter_edge_size`, preserving aspect ratio."""
        h, w = image_hwc_f32.shape[:2]
        if w < h:
            new_w = shorter_edge_size
            new_h = int(h * shorter_edge_size / w)
        else:
            new_h = shorter_edge_size
            new_w = int(w * shorter_edge_size / h)
        return jax.image.resize(image_hwc_f32, (new_h, new_w, image_hwc_f32.shape[-1]), method='bicubic')

    def _jax_center_crop(self, image_hwc_f32: jnp.ndarray, crop_size: int) -> jnp.ndarray:
        """Center crops an image (HWC, self.dtype) to `crop_size` x `crop_size`."""
        h, w = image_hwc_f32.shape[:2]
        th = tw = crop_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return image_hwc_f32[i:i+th, j:j+tw, :]

    def _jax_to_chw_tensor(self, image_hwc_f32: jnp.ndarray) -> jnp.ndarray:
        """Converts an image from HWC, self.dtype [0,1] to CHW, self.dtype [0,1]."""
        if image_hwc_f32.dtype == jnp.uint8: # Ensure float [0,1]
            image_hwc_f32 = image_hwc_f32.astype(jnp.float32) / 255.0
        return jnp.transpose(image_hwc_f32, (2, 0, 1))

    def _jax_normalize(self, image_chw_f32: jnp.ndarray, mean: tuple, std: tuple) -> jnp.ndarray:
        """Normalizes a CHW, self.dtype image tensor."""
        mean_jnp = jnp.array(mean, dtype=self.dtype).reshape(image_chw_f32.shape[0], 1, 1)
        std_jnp = jnp.array(std, dtype=self.dtype).reshape(image_chw_f32.shape[0], 1, 1)
        return (image_chw_f32 - mean_jnp) / (std_jnp + 1e-8) # Epsilon for stability


class FlaxMLP(nn.Module):
    """Flax implementation of the MLP from ImageReward."""
    input_size: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x_input: jnp.ndarray, deterministic: bool = True):
        dropout_rng = None
        if not deterministic:
            dropout_rng = self.make_rng("dropout")
            
        x = nn.Dense(features=1024, name="fc1", dtype=self.dtype,
                     kernel_init=jax.nn.initializers.normal(stddev=1.0 / (self.input_size + 1)),
                     bias_init=jax.nn.initializers.zeros)(x_input)
        x = nn.Dropout(0.2, deterministic=deterministic)(x)
        x = nn.Dense(features=128, name="fc2", dtype=self.dtype,
                     kernel_init=jax.nn.initializers.normal(stddev=1.0 / (1024 + 1)),
                     bias_init=jax.nn.initializers.zeros)(x)
        x = nn.Dropout(0.2, deterministic=deterministic)(x)
        x = nn.Dense(features=64, name="fc3", dtype=self.dtype,
                     kernel_init=jax.nn.initializers.normal(stddev=1.0 / (128 + 1)),
                     bias_init=jax.nn.initializers.zeros)(x)
        x = nn.Dropout(0.1, deterministic=deterministic)(x)
        x = nn.Dense(features=16, name="fc4", dtype=self.dtype,
                     kernel_init=jax.nn.initializers.normal(stddev=1.0 / (64 + 1)),
                     bias_init=jax.nn.initializers.zeros)(x)
        x = nn.Dense(features=1, name="fc5", dtype=self.dtype,
                     kernel_init=jax.nn.initializers.normal(stddev=1.0 / (16 + 1)),
                     bias_init=jax.nn.initializers.zeros)(x)
        return x

class FlaxIRSMC(nn.Module):
    med_config: str
    mean: float = 0.16717362830052426  # from ImageReward
    std: float = 1.0333394966054072  # from ImageReward
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blip = FlaxBLIP_Pretrain(
            image_size=224,
            vit='base', 
            med_config=self.med_config,
            dtype=self.dtype
        )
        self.preprocess = FlaxImageRewardPreprocessor(n_px=224, dtype=self.dtype)
        self.mlp = FlaxMLP(768, dtype=self.dtype)

    def __call__(
        self,
        prompts, images,
        deterministic=True
    ):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        images = [self.preprocess(image) for image in images]
        images = jnp.stack(images, axis=0)  # (B, C, H, W)

        txt_features = self.blip(images=images, prompts=prompts, deterministic=deterministic)

        rewards = self.mlp(txt_features, deterministic=deterministic)
        rewards = (rewards - self.mean) / self.std

        return rewards

def rm_load(
    name: str = "ImageReward-v1.0",
    dtype: jnp.dtype = jnp.float32,
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward Flax model 
    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    dtype : jax.numpy.dtype
        The data type of the computation

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    med_config: str
        path for the mixture of encoder-decoder model's configuration file

    Returns
    -------
    model : flax.linen.Module
        The ImageReward model
    """
    if name in RM.utils._MODELS:
        model_path = RM.ImageReward_download(
            RM.utils._MODELS[name],
            download_root or os.path.expanduser("~/.cache/ImageReward"),
        )
        # TODO: Convert .pt weights to be compatible with Flax
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")

    # med_config
    if med_config is None:
        med_config = RM.ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root or os.path.expanduser("~/.cache/ImageReward"),
        )

    model = FlaxIRSMC(med_config=med_config, dtype=dtype)
    # TODO: convert torch state dict and setup flax model

    return model