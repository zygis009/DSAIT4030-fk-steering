# fkd_class_jax.py  ───────────────────────────────────────────────────────────
from enum import Enum
import numpy as np
import jax.numpy as jnp
import jax


class PotentialType(str, Enum):
    DIFF = "diff"
    MAX  = "max"
    ADD  = "add"


class FKD:
    """
    JAX implementation of the Feynman-Kac Diffusion (FKD) steering mechanism.

    Call `.resample()` at each reverse-diffusion step.
    """

    def __init__(
        self,
        *,
        potential_type      = PotentialType.DIFF,
        lmbda               : float,
        num_particles       : int,
        adaptive_resampling : bool,
        resample_frequency  : int,
        resampling_t_start  : int,
        resampling_t_end    : int,
        time_steps          : int,
        reward_fn,                     # images → np.float32[N] or jnp.ndarray
        reward_min_value    : float = 0.,
        latent_to_decode_fn = lambda x: x,   # latents → images
    ):
        self.ptype  = PotentialType(potential_type)
        self.lmbda  = float(lmbda)
        self.N      = int(num_particles)

        self.adapt  = adaptive_resampling
        self.freq   = int(resample_frequency)
        self.t0     = int(resampling_t_start)
        self.t1     = int(resampling_t_end)
        self.timesteps      = int(time_steps)

        self.reward_fn           = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        self.population_r       = jnp.ones(self.N) * reward_min_value
        self.prod_potentials    = jnp.ones(self.N)

    # Resample using a reward model and potential function
    def resample(
        self, *,
        sampling_idx: int,
        latents: jnp.ndarray,
        x0_preds: jnp.ndarray,
    ):
        """Return (new_latents, None)."""

        # Decide whether this timestep is *eligible* for resampling
        allowed = np.append(
            np.arange(self.t0, self.t1 + 1, self.freq),
            self.timesteps - 1,
        )
        if sampling_idx not in allowed:
            return latents, None

        # Compute rewards
        imgs     = self.latent_to_decode_fn(x0_preds)
        rewards  = jnp.asarray(self.reward_fn(imgs), dtype=jnp.float32)

        # Convert potential to un-normalised weights  
        if self.ptype == PotentialType.MAX:
            w = jnp.exp(self.lmbda * jnp.maximum(rewards, self.population_r))
        elif self.ptype == PotentialType.ADD:
            w = jnp.exp(self.lmbda * (rewards + self.population_r))
        else:  # DIFF
            w = jnp.exp(self.lmbda * (rewards - self.population_r))

        # Special final-step correction for MAX/ADD
        if sampling_idx == self.timesteps - 1 and self.ptype in (PotentialType.MAX, PotentialType.ADD):
            w = jnp.exp(self.lmbda * rewards) / self.prod_potentials

        # Clamp / normalise
        w = jnp.clip(w, 0.0, 1e10)
        p = w / w.sum()

        # Adaptive resampling via ESS
        do_resample = True
        if self.adapt and sampling_idx != self.timesteps - 1:
            ess = 1.0 / jnp.sum(p ** 2)
            do_resample = ess < 0.5 * self.N   # same threshold as paper

        if do_resample:
            key = jax.random.PRNGKey(sampling_idx + 1234)
            idx = jax.random.choice(key, self.N, (self.N,), replace=True, p=p)

            latents  = latents[idx]
            rewards  = rewards[idx]
            self.prod_potentials = self.prod_potentials[idx] * w[idx]
        # else: keep latents untouched, just update potentials state
        else:
            self.prod_potentials = self.prod_potentials * w

        self.population_r = rewards
        return latents, None