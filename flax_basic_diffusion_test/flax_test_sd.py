import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import matplotlib.pyplot as plt
import time
from diffusers import FlaxStableDiffusionPipeline
from diffusers import FlaxDPMSolverMultistepScheduler

# Memory management config
jax.config.update('jax_platform_name', 'gpu')
print(f"Default device for inference: {jax.default_backend()}")  

# Configure DPM Solver with Karras noise schedule
scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)
scheduler.config.use_karras_sigmas = True 

# Load SD pipeline with the configured scheduler
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    revision="bf16",
    dtype=jax.numpy.bfloat16
)
params["scheduler"] = scheduler_state 

# Config experiment
prompt = "a photo of an astronaut riding a horse on mars"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

# Prepare inputs
num_samples = 1
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# Shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, num_samples)
prompt_ids = shard(prompt_ids)

# Initial run to load model parameters into GPU memory
print("Running initial model parameters into GPU memory...")
_ = pipeline(prompt_ids, params, prng_seed, 1, jit=True)  # Use 1 step to load model parameters
print("Model loaded into GPU memory. Starting generation...")

# Time the second generation
start_time = time.time()
images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
generation_time = time.time() - start_time
print(f"Image generation took: {generation_time:.2f} seconds (after initial load)")

# Process and display images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
if images:
    images[0].save("output.png")
    plt.imshow(images[0])
    plt.axis('off')
    plt.show()