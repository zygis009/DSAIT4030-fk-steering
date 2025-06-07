from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import CLIPImageProcessor, CLIPTokenizerFast, FlaxCLIPModel
from PIL import Image
from typing import Tuple, Dict, Union


class FlaxCLIPScore(nn.Module):
    """
    A Flax Linen module for calculating CLIP scores.
    """
    model_name: str = "openai/clip-vit-large-patch14"
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """
        Initializes the FlaxCLIPScore submodules
        """
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self.clip = FlaxCLIPModel.from_pretrained(self.model_name, dtype=self.dtype)

    def __call__(self, prompts: Union[str, jnp.ndarray], images: Union[str, Image.Image, jnp.ndarray], return_feature: bool = False) -> Union[float, Tuple[float, Dict[str, jnp.ndarray]]]:
        """
        Calculates the CLIP score between a prompt and an image.

        Args:
            prompt: The text prompt (string) or pre-tokenized input_ids (jnp.ndarray).
            image: A PIL Image, string path to image or jnp.ndarray postprocessed pixel values.
            return_feature: whether to return image and text features along with the score.

        Returns:
            The CLIP score, or a tuple containing the score and a dictionary of features
            if return_feature is True.
        """
        
        # text encoding
        if isinstance(prompts, str):
            input_ids = self.tokenizer(
                prompts,
                return_tensors="jax",
                truncation=True,
            ).input_ids
        elif isinstance(prompts, jnp.ndarray):
            input_ids = prompts
        else:
            raise ValueError(f"Prompt needs to be of type `str` or `jnp.ndarray`. Got {type(prompts)}")
        text_features = self.clip.get_text_features(input_ids)

        # Image encoding
        if isinstance(images, Image.Image):
            pixel_values = self.image_processor(images=[images], return_tensors="jax").pixel_values
        elif isinstance(images, str):
            pixel_values = self.image_processor(images=[Image.open(images)], return_tensors="jax").pixel_values
        elif isinstance(images, jnp.ndarray):
            # define params for jnp.ndarray image preprocessing
            CLIP_CROP_SIZE = 224
            OPENAI_CLIP_MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073], dtype=self.dtype)
            OPENAI_CLIP_STD = jnp.array([0.26862954, 0.26130258, 0.27577711], dtype=self.dtype)
            pixel_values = jax_clip_preprocess(images, CLIP_CROP_SIZE, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
        else:
            raise ValueError(f"Image needs to be of type `Pil.Image.Image`, `str` or `jnp.ndarray`. Got {type(images)}")
        
        image_features = self.clip.get_image_features(pixel_values)

        rewards = jnp.sum(jnp.multiply(text_features, image_features), axis=1, keepdims=True)

        if return_feature:
            return rewards.squeeze(), {'image': image_features, 'txt': text_features}

        return rewards.squeeze()

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def jax_clip_preprocess(
    images: jnp.ndarray,
    target_size: int = 224,
    latent_means: jnp.ndarray = jnp.array([0.48145466, 0.4578275, 0.40821073], dtype=jnp.float32),
    latent_stds: jnp.ndarray = jnp.array([0.26862954, 0.26130258, 0.27577711], dtype=jnp.float32),
    do_resize: bool = True,
    do_center_crop: bool = True,
    do_normalize: bool = True,
    data_format: str = 'channels_first'
) -> jnp.ndarray:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Input images must be of shape (batch, H, W, C) with C=3.")

    images = jnp.transpose(images, (0, 2, 3, 1))

    if do_resize:
        images = jax.image.resize(
            images,
            shape=(images.shape[0], target_size, target_size, images.shape[-1]),
            method='bilinear'
        )

    if do_center_crop and (images.shape[1] > target_size or images.shape[2] > target_size):
        h, w = images.shape[1:3]
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        images = jax.lax.dynamic_slice(
            images,
            start_indices=(0, start_h, start_w, 0),
            slice_sizes=(images.shape[0], target_size, target_size, images.shape[-1])
        )

    if do_normalize:
        mean_reshaped = latent_means.reshape((1, 1, 1, -1))
        std_reshaped = latent_stds.reshape((1, 1, 1, -1))
        images = (images - mean_reshaped) / std_reshaped


    if data_format == 'channels_first':
        images = jnp.transpose(images, (0, 3, 1, 2)) # (N,C,H,W)
    elif data_format == 'channels_last':
        pass
    else:
        raise ValueError(f"Unsupported data_format: {data_format}. Must be 'channels_first' or 'channels_last'.")

    return images