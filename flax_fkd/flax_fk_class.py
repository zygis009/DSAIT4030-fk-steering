import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap

from typing import Callable, NamedTuple, Optional, Tuple
from enum import Enum

class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"

# JAX pattern for managing changing variables:
class FlaxFKDState(NamedTuple):
    population_rs: jnp.ndarray
    product_of_potentials: jnp.ndarray
    prng_key: jax.random.PRNGKey

class FlaxFKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[jnp.ndarray], jnp.ndarray],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.last_sampling_idx = time_steps - 1  # slightly reduces future number of operations
        
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        self.reward_min_value = reward_min_value
        
        # Determine the fixed interval for resampling checks (can be done once)
        self._resampling_interval = jnp.append(jnp.arange(
            resampling_t_start, resampling_t_end + 1, resample_frequency
        ), self.last_sampling_idx)

    def init_state(self, prng_key: jax.random.PRNGKey) -> FlaxFKDState:
        return FlaxFKDState(
            population_rs=jnp.ones(self.num_particles, dtype=jnp.float32) * self.reward_min_value,
            product_of_potentials=jnp.ones(self.num_particles, dtype=jnp.float32),
            prng_key=prng_key
        )
    
    def resample(self,
                 *,
                 current_fkd_state: FlaxFKDState,
                 sampling_idx: int, 
                 latents: jnp.ndarray, 
                 x0_preds: jnp.ndarray,
                 
    ) -> Tuple[FlaxFKDState, jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            current_fkd_state: Current dynamic state of FKD.
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing new FKD state, resampled latents and optionally resampled images.
        """

        def _no_resample_branch():
            return current_fkd_state, latents, None # No change in latents or FKD state, no images resampled
        
        def _resample_branch():
            key, resample_key, choice_key = jax.random.split(current_fkd_state.prng_key, 3)
            
            # Decode latents to population images and compute rewards
            population_images = self.latent_to_decode_fn(x0_preds)
            rs_candidates = self.reward_fn(population_images)
                        
            # Compute importance weights
            if self.potential_type == PotentialType.MAX:
                w = jnp.exp(self.lmbda * jnp.maximum(rs_candidates, current_fkd_state.population_rs))
            elif self.potential_type == PotentialType.ADD:
                rs_candidates = rs_candidates + current_fkd_state.population_rs
                w = jnp.exp(self.lmbda * rs_candidates)
            elif self.potential_type == PotentialType.DIFF:
                diffs = rs_candidates - current_fkd_state.population_rs
                w = jnp.exp(self.lmbda * diffs)
            else:
                raise ValueError(f"potential_type {self.potential_type} not recognized")

            if self.potential_type in (PotentialType.MAX, PotentialType.ADD):
                w = jax.lax.cond(
                    sampling_idx == self.last_sampling_idx,
                    lambda: jnp.exp(self.lmbda * rs_candidates) / current_fkd_state.product_of_potentials,
                    lambda: w
                )
    
            w = jnp.nan_to_num(jnp.clip(w, 0, 1e10), nan=0.0)
            jax.debug.print("Candidates: {rs_candidates}; weights: {w}", rs_candidates=rs_candidates, w=w)
            # return current_fkd_state, latents, None
            adaptive_resample = self.adaptive_resampling | (sampling_idx == self.last_sampling_idx)

            normalized_w = w / jnp.sum(w)
            ess = 1.0 / jnp.sum(jnp.power(normalized_w, 2))
            
            def _do_resampling(w_val, rs_candidates_val):
                # Resample indices based on weights
                logits = jax.nn.log_softmax(w_val, axis=-1)
                indices = jax.random.choice(
                    choice_key,
                    a=jnp.arange(self.num_particles),
                    shape=(self.num_particles,),
                    p=logits,
                    replace=True,
                )

                resampled_latents = latents[indices]
                new_population_rs = rs_candidates_val[indices]
                resampled_images = population_images[indices]
                new_product_of_potentials = (
                    current_fkd_state.product_of_potentials[indices] * w_val[indices]
                )
                return new_population_rs, new_product_of_potentials, resampled_latents, resampled_images
                
                
            def _no_resampling(w_val, rs_candidates_val):
                # No resampling
                new_population_rs = rs_candidates_val
                new_product_of_potentials = current_fkd_state.product_of_potentials
                resampled_latents_out = latents
                resampled_images_out = population_images

                return new_population_rs, new_product_of_potentials, resampled_latents_out, resampled_images_out

            new_population_rs, new_product_of_potentials, resampled_latents, resampled_images = jax.lax.cond(
                adaptive_resample,
                lambda: jax.lax.cond(
                    ess < 0.5 * self.num_particles, # ess check
                    lambda: _do_resampling(w, rs_candidates), # true -> perform resampling
                    lambda: _no_resampling(w, rs_candidates), # false -> no resampling
                ),
                lambda: _do_resampling(w, rs_candidates), # Always resample if not adaptive or last_sampling_idx
            )
            
            new_fkd_state = FlaxFKDState(
                population_rs=new_population_rs, 
                product_of_potentials=new_product_of_potentials, 
                prng_key=key
            )
            return new_fkd_state, resampled_latents, resampled_images

        # Check if resampling interval condition is met
        return jax.lax.cond(
            jnp.any(self._resampling_interval == sampling_idx),
            _resample_branch,
            _no_resample_branch
        )