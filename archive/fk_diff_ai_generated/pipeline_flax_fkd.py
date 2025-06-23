# pipeline_flax_fkd.py

import jax
import jax.numpy as jnp
# ▼▼▼ ADD THIS IMPORT ▼▼▼
import jax.debug
from jax.experimental.host_callback import call
from typing import Optional, List, Dict, Any, Union
from functools import partial

from diffusers import FlaxStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    FlaxStableDiffusionPipelineOutput,
)
from flax.core.frozen_dict import FrozenDict

from fkd_flax import FKDState, init_fkd_state, fkd_resample, PotentialType
from rewards_flax import get_reward_scores

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class FlaxFKDStableDiffusionPipeline(FlaxStableDiffusionPipeline):
    """
    A Flax Stable Diffusion Pipeline with integrated Feynman-Kac (FK) steering.
    """

    def _get_pred_original_sample(self, scheduler_params, latents, t, noise_pred):
        """Calculates x0 from xt and noise, as FlaxDDIMScheduler.step doesn't return it."""
        alpha_prod_t = scheduler_params.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            latents - beta_prod_t ** (0.5) * noise_pred
        ) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def _fkd_generate(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        fkd_args: Dict[str, Any],
        prompts: List[str],
    ):
        # 1. Setup
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8.")
            
        num_particles = prompt_ids.shape[0]
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        uncond_input = self.tokenizer(
            [""] * num_particles, padding="max_length", max_length=prompt_ids.shape[-1], return_tensors="np"
        ).input_ids
        negative_prompt_embeds = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
        context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

        latents_key, fkd_key, loop_key = jax.random.split(prng_seed, 3)

        latents_shape = (num_particles, self.unet.config.in_channels, height // 8, width // 8)
        latents = jax.random.normal(latents_key, shape=latents_shape, dtype=self.dtype)

        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"], num_inference_steps=num_inference_steps, shape=latents.shape
        )
        latents = latents * scheduler_state.init_noise_sigma

        fkd_state = init_fkd_state(fkd_key, num_particles, reward_min_value=fkd_args.get("reward_min_value", 0.0))
        
        # ▼▼▼ ADD THIS DEBUG PRINT ▼▼▼
        # This will run once before the loop starts.
        jax.debug.print("--- [Diagnostic] `params['scheduler']` before loop: {x}", x=params["scheduler"])


        # 2. Denoising loop
        def loop_body(i, carry):
            latents, fkd_state, scheduler_state, key = carry
            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[i]

            latent_model_input = jnp.concatenate([latents] * 2)
            scaled_latent_model_input = self.scheduler.scale_model_input(scheduler_state, latent_model_input, t)

            noise_pred = self.unet.apply(
                {"params": params["unet"]}, scaled_latent_model_input, t, context
            ).sample

            noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- FKD STEERING LOGIC ---
            # a. Get predicted x0

            # ▼▼▼ ADD THIS DEBUG PRINT ▼▼▼
            # This will run on every step of the loop.
            jax.debug.print("--- [Diagnostic] `params['scheduler']` in loop step {i}: {x}", i=i, x=params["scheduler"])
            
            x0_pred_latents = self._get_pred_original_sample(
                params["scheduler"], latents, t, noise_pred
            )
            
            # ... (rest of the code is the same)
            x0_pred_latents_scaled = 1 / self.vae.config.scaling_factor * x0_pred_latents
            image = self.vae.apply({"params": params["vae"]}, x0_pred_latents_scaled, method=self.vae.decode).sample
            image_uint8 = ((image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1) * 255).astype(jnp.uint8)

            rewards = call(
                lambda img: get_reward_scores(img, prompts, fkd_args["guidance_reward_fn"]),
                image_uint8,
                result_shape_and_dtype=jax.ShapeDtypeStruct((num_particles,), jnp.float32),
            )

            step_output = self.scheduler.step(scheduler_state, noise_pred, t, latents)
            next_latents = step_output.prev_sample
            scheduler_state = step_output.state

            resampled_latents, fkd_state = fkd_resample(
                fkd_state,
                next_latents,
                rewards,
                fkd_args["lmbda"],
                PotentialType(fkd_args["potential_type"]),
            )

            return resampled_latents, fkd_state, scheduler_state, key

        # Run the loop
        if DEBUG:
            carry = (latents, fkd_state, scheduler_state, loop_key)
            for i in range(num_inference_steps):
                carry = loop_body(i, carry)
            final_latents, _, _, _ = carry
        else:
            final_latents, _, _, _ = jax.lax.fori_loop(
                0, num_inference_steps, loop_body, (latents, fkd_state, scheduler_state, loop_key)
            )

        # 3. Post-process
        latents = 1 / self.vae.config.scaling_factor * final_latents
        image = self.vae.apply(
            {"params": params["vae"]}, latents, method=self.vae.decode
        ).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)

        return image
    
    # The __call__ method remains unchanged
    def __call__(
        self,
        prompt_ids: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.Array,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 7.5,
        fkd_args: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        return_dict: bool = True,
        jit: bool = False,
    ):
        if fkd_args is None:
            return super().__call__(
                prompt_ids=prompt_ids,
                params=params,
                prng_seed=prng_seed,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                return_dict=return_dict,
                jit=jit,
            )

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if jit:
            generate_fn = partial(
                self._fkd_generate,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                fkd_args=fkd_args,
                prompts=prompts,
            )
            images = jax.jit(generate_fn, static_argnums=(3,))(
                prompt_ids, params, prng_seed, num_inference_steps
            )
        else:
            images = self._fkd_generate(
                prompt_ids,
                params,
                prng_seed,
                num_inference_steps,
                height,
                width,
                guidance_scale,
                fkd_args,
                prompts,
            )

        has_nsfw_concept = [False] * images.shape[0]

        if not return_dict:
            return (images, has_nsfw_concept)

        return FlaxStableDiffusionPipelineOutput(
            images=images, nsfw_content_detected=has_nsfw_concept
        )