"""
Flax Stable-Diffusion 1.4 pipeline **with FK-Steering**

This file subclasses the stock `FlaxStableDiffusionPipeline` from Diffusers and
injects the Feynman-Kac Diffusion (FKD) resampling logic

*   `_generate` is copied from Diffusers and augmented where necessary.
*   `__call__` is reimplemented so we can pass the FKD-specific keyword
    arguments without upsetting the upstream signature and to guarantee that
    jitting/pmap decisions remain under our control.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from diffusers import FlaxStableDiffusionPipeline as _Base
from diffusers import FlaxDDIMScheduler
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput

from fkd_class import FKD, PotentialType
from rewards import get_reward_function
from tqdm.auto import tqdm 

# ──────────────────────────────────────────────────────────────────────────────
# Helper: decode a batch of latent tensors on host to a list of PIL images.
# -----------------------------------------------------------------------------


def _latents_to_pil(pipe: _Base, lat_bf16: jnp.ndarray) -> List[Image.Image]:
    arr = np.asarray(lat_bf16, dtype=np.float32)
    imgs = pipe.decode_latents(arr)
    return [Image.fromarray(i) for i in imgs]


# ──────────────────────────────────────────────────────────────────────────────
# Main subclass
# -----------------------------------------------------------------------------


class FKDFlaxStableDiffusion(_Base):
    """Stable‑Diffusion‑1·4 in Flax with optional FK‑Steering."""

    # ------------------------------------------------------------------
    # Utility: decode latents on *host* in the same way the pipeline does
    # when returning the final images.  We keep it a method so FKD can
    # call it via `_latents_to_pil`.
    # ------------------------------------------------------------------
    def decode_latents(self, latents: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_params"):
            raise AttributeError("Pipeline parameters not set; decode_latents "
                                 "called outside an active generation.")
        latents = jnp.asarray(latents, dtype=jnp.float32) / self.vae.config.scaling_factor
        imgs = self.vae.apply({"params": self._params["vae"]}, latents, method=self.vae.decode).sample
        imgs = (imgs / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)  # (B, H, W, 3)
        imgs = np.asarray(imgs * 255).round().astype("uint8")
        return imgs

    """Stable‑Diffusion‑1·4 in Flax with optional FK‑Steering."""

    # ------------------------------------------------------------------
    # Public entry point.  We *do not* call `super().__call__` because we
    # want full control over the signature and because we funnel the
    # entire generation through our patched `_generate` implementation.
    # ------------------------------------------------------------------
    def __call__(
        self,
        prompt_ids,
        params: Dict[str, Any],
        prng_seed,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        latents: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
        *,
        fkd_args: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        jit: Optional[bool] = None,
    ) -> FlaxStableDiffusionPipelineOutput:
        """Generate images with (optional) FK‑Steering.

        All arguments up to ``neg_prompt_ids`` mirror the upstream
        pipeline.  ``fkd_args`` and ``prompts`` are new and only used when
        you enable FKD.
        """

        # FK‑Steering relies on a Python loop over the time‑steps, which is
        # *not* compatible with the standard `jax.jit`‑compiled version of
        # the pipeline.  We therefore force `jit=False` whenever FKD is
        # requested, unless the caller explicitly overrides it with
        # `jit=True` (not recommended).
        if fkd_args and fkd_args.get("use_smc", False):
            jit = False if jit is None else jit

        # Stash params so `decode_latents` can access them from FKD helpers
        self._params = params

        images = self._generate(
            prompt_ids,
            params,
            prng_seed,
            num_inference_steps,
            height,
            width,
            guidance_scale,
            latents,
            neg_prompt_ids,
            fkd_args=fkd_args,
            prompts=prompts,
        )

       #  Safety checker (mandatory according to guidelines...)
        has_nsfw = False                        
        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            imgs_uint8 = np.asarray((images * 255).round(), dtype="uint8")
            flat_imgs  = imgs_uint8.reshape((-1, *imgs_uint8.shape[-3:]))
            flat_imgs, has_nsfw = self._run_safety_checker(flat_imgs, safety_params, jit)
            imgs_uint8 = flat_imgs.reshape(imgs_uint8.shape)
            images     = imgs_uint8.astype(np.float32) / 255.0
        else:
            has_nsfw = False

        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw)
    # ------------------------------------------------------------------
    # based on the original Diffusers implementation.  We insert FKD resampling.
    # ------------------------------------------------------------------
    def _generate(
        self,
        prompt_ids: jnp.ndarray,
        params: Dict[str, Any],
        prng_seed: jax.Array,
        num_inference_steps: int,
        height: int,
        width: int,
        guidance_scale: float,
        latents: Optional[jnp.ndarray] = None,
        neg_prompt_ids: Optional[jnp.ndarray] = None,
        *,
        fkd_args: Optional[Dict[str, Any]],
        prompts: Optional[List[str]],
    ) -> jnp.ndarray:

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        # get prompt text embeddings
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size   = prompt_ids.shape[0]
        max_length   = prompt_ids.shape[-1]

        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="np",
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
        negative_prompt_embeds = self.text_encoder(
            uncond_input, params=params["text_encoder"]
        )[0]
        context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])

        # Ensure model output will be `float32` before going into the scheduler
        guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)

        # 2.  Latent tensor initialisation
        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width  // self.vae_scale_factor,
        )

        if latents is None:
            # generate *independent* x_T for each particle
            keys = jax.random.split(prng_seed, batch_size)

            def _sample(rng):
                return jax.random.normal(rng, latents_shape[1:], dtype=jnp.float32)

            latents = jax.vmap(_sample)(keys)        # shape = latents_shape
        else:
            if latents.shape != latents_shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
                )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * params["scheduler"].init_noise_sigma

        # FK-steering implementation
        if fkd_args and fkd_args.get("use_smc", False):
            def _reward(pil_imgs):
                return get_reward_function(
                    fkd_args["guidance_reward_fn"],
                    images=pil_imgs,
                    prompts=prompts,
                    metric_to_chase=fkd_args.get("metric_to_chase"),
                )

            fkd = FKD(
                potential_type      = fkd_args.get("potential_type", "diff"),
                lmbda               = fkd_args.get("lmbda", 10.0),
                num_particles       = batch_size,
                adaptive_resampling = fkd_args.get("adaptive_resampling", False),
                resample_frequency  = fkd_args.get("resample_frequency", 1),
                resampling_t_start  = -1,
                resampling_t_end    = num_inference_steps,
                time_steps          = num_inference_steps,
                reward_fn           = _reward,
                latent_to_decode_fn = lambda z: _latents_to_pil(self, z),
            )
        else:
            fkd = None

        # 4.  Scheduler initial state
        scheduler_state = self.scheduler.set_timesteps(
            params["scheduler"],
            num_inference_steps=num_inference_steps,
            shape=latents_shape,
        )

        # 5.  Reverse-diffusion loop -------------------------------------------
        for step in tqdm(range(num_inference_steps),
                        desc="FK-Steering",
                        unit="step"):
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            latents_input = jnp.concatenate([latents] * 2)

            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            timestep = jnp.broadcast_to(t, latents_input.shape[0])

            latents_input = self.scheduler.scale_model_input(
                scheduler_state, latents_input, t
            )

            # predict the noise residual
            noise_pred = self.unet.apply(
                {"params": params["unet"]},
                jnp.array(latents_input),
                jnp.array(timestep, dtype=jnp.int32),
                encoder_hidden_states=context,
            ).sample
            # perform guidance
            noise_pred_uncond, noise_prediction_text = jnp.split(noise_pred, 2, axis=0)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_prediction_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            step_out = self.scheduler.step(
                scheduler_state, noise_pred, t, latents, return_dict=True
            )
            latents, scheduler_state = step_out.prev_sample, step_out.state
            
            # Force x0 to being the original sample
            if (
                hasattr(step_out, "pred_original_sample")
                and step_out.pred_original_sample is not None
            ):
                x0_pred = step_out.pred_original_sample
            else:
                # Older Diffusers versions (or certain schedulers) don't
                # return x0; fall back to the current latent estimate.
                # This is sufficient for FKD weighting because resampling
                # only relies on **relative** rewards.
                x0_pred = latents

            # RESAMPLE USING FKD
            if fkd is not None:
                latents, _ = fkd.resample(
                    sampling_idx=step, latents=latents, x0_preds=x0_pred
                )
                
        # 6.  Decode latents ----------------------------------------------------
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample

        images = (images / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return images