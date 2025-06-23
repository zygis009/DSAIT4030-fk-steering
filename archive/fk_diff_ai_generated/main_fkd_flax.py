# main_fkd_flax.py

import jax
import numpy as np
import time
import matplotlib.pyplot as plt
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxDDIMScheduler

# Import our custom pipeline and reward helpers
from pipeline_flax_fkd import FlaxFKDStableDiffusionPipeline
from rewards_flax import load_reward_models

# --- 1. Configuration ---
MODEL_ID = "CompVis/stable-diffusion-v1-4"
DTYPE = jax.numpy.bfloat16

# FKD Steering configuration
NUM_PARTICLES = 4  # Number of images to generate and resample from.
FKD_ARGS = {
    "guidance_reward_fn": "ImageReward",  # or "HumanPreference", "Clip-Score"
    "potential_type": "diff",             # or "max", "add"
    "lmbda": 10.0,                        # Steering strength
    "guidance_scale": 7.5                 # Standard CFG scale
}

PROMPT = "a picture of a brown knife and a blue donut"
NUM_INFERENCE_STEPS = 50
SEED = 42

# --- 2. Setup Models ---
# Load PyTorch reward models on the host (CPU/GPU) *before* JIT compilation
print("Loading PyTorch-based reward models...")
load_reward_models(device='cuda')

# Load the Flax pipeline
scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipeline, params = FlaxFKDStableDiffusionPipeline.from_pretrained(
    MODEL_ID, scheduler=scheduler, revision="bf16", dtype=DTYPE
)
params["scheduler"] = scheduler_state

# --- 3. Prepare Inputs for JAX ---
prng_seed = jax.random.PRNGKey(SEED)
prompts = [PROMPT] * NUM_PARTICLES
prompt_ids = pipeline.prepare_inputs(prompts)

# Replicate for multi-device execution if available (optional but good practice)
# For simplicity, we run on a single device if only one is available.
if jax.device_count() > 1:
    params = replicate(params)
    prng_seed = jax.random.split(prng_seed, jax.device_count())
    prompt_ids = shard(prompt_ids)

# --- 4. Run Generation ---
print("Starting image generation with FKD steering...")
start_time = time.time()

# Call our custom pipeline with fkd_args
images = pipeline(
    prompt_ids=prompt_ids,
    params=params,
    prng_seed=prng_seed,
    num_inference_steps=NUM_INFERENCE_STEPS,
    fkd_args=FKD_ARGS,
    prompts=prompts,  # Pass prompts for the reward function
    jit=True
).images

# Ensure results are back on the host
images = np.asarray(images.block_until_ready())
generation_time = time.time() - start_time
print(f"Image generation took: {generation_time:.2f} seconds")

# --- 5. Display Results ---
pil_images = pipeline.numpy_to_pil(images)
fig, axs = plt.subplots(1, NUM_PARTICLES, figsize=(20, 5))
fig.suptitle(f"Generated Images with {FKD_ARGS['guidance_reward_fn']} Steering (lmbda={FKD_ARGS['lmbda']})")
for i, img in enumerate(pil_images):
    if NUM_PARTICLES > 1:
        ax = axs[i]
    else:
        ax = axs
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.savefig("output_fkd_flax.png")
print("Saved generated images to output_fkd_flax.png")
plt.show()