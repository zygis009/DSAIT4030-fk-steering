# fkd_flax.py

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from enum import Enum

class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"

@dataclass
class FKDState:
    """State for the FKD steering mechanism."""
    population_rs: jax.Array
    product_of_potentials: jax.Array
    key: jax.random.PRNGKey

def init_fkd_state(key: jax.random.PRNGKey, num_particles: int, reward_min_value: float = 0.0) -> FKDState:
    """Initializes the FKD state."""
    key, subkey = jax.random.split(key)
    return FKDState(
        population_rs=jnp.ones(num_particles, dtype=jnp.float32) * reward_min_value,
        product_of_potentials=jnp.ones(num_particles, dtype=jnp.float32),
        key=subkey,
    )

def fkd_resample(
    state: FKDState,
    latents: jax.Array,
    rewards: jax.Array,
    lmbda: float,
    potential_type: PotentialType,
) -> tuple[jax.Array, FKDState]:
    """Performs the FKD resampling step in a functional way."""
    key, resample_key = jax.random.split(state.key)

    if potential_type == PotentialType.MAX:
        w = jnp.exp(lmbda * jnp.maximum(rewards, state.population_rs))
    elif potential_type == PotentialType.ADD:
        w = jnp.exp(lmbda * (rewards + state.population_rs))
    elif potential_type == PotentialType.DIFF:
        w = jnp.exp(lmbda * (rewards - state.population_rs))
    else:
        raise ValueError(f"potential_type {potential_type} not recognized")

    w = jnp.clip(w, 0, 1e10)
    w = jnp.nan_to_num(w)

    indices = jax.random.multinomial(resample_key, w, shape=(latents.shape[0],), replacement=True)
    resampled_latents = latents[indices]
    
    new_state = state.replace(
        population_rs=rewards[indices],
        product_of_potentials=state.product_of_potentials[indices] * w[indices],
        key=key,
    )
    return resampled_latents, new_state