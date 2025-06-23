# demo_fkd_flax.py  ───────────────────────────────────────────────────────────
import os
import jax
import numpy as np
from diffusers import FlaxDDIMScheduler
from fkd_pipeline_sd14_flax import FKDFlaxStableDiffusion
from plot_helper import save_image_grid, create_output_dir, config_hash
import json

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())

# 1.  Load pipeline & scheduler (single-GPU)
scheduler, sched_state = FlaxDDIMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler", dtype=jax.numpy.float16
)
pipe, params = FKDFlaxStableDiffusion.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    scheduler=scheduler,
    revision="bf16",                 # keep weights in bfloat16
    dtype=jax.numpy.float16,
    safety_checker=None,          #  disables safety checker (faster inference)
)
params["scheduler"] = sched_state            # attach scheduler state

# 2.  Experiment configuration
NUM_PARTICLES   = 2
TIME_STEPS      = 100        # paper uses 100 DDIM steps
PROMPT          = "a photo of a brown knife and a blue donut"

fkd_cfg = dict(
    use_smc            = True,
    potential_type     = "max",        # or "diff", "add"
    lmbda              = 10,          # paper default
    num_particles      = NUM_PARTICLES,
    time_steps         = TIME_STEPS,
    adaptive_resampling= True,
    resample_frequency = 20,    # paper uses 20
    resampling_t_start = 20,    # paper uses 20
    resampling_t_end   = 80,    # paper uses 80
    guidance_reward_fn = "ImageReward",
    metric_to_chase    = None,         # only needed with "LLMGrader"
)

# 3.  Deterministic seeding  (JAX)
SEED = 42
prng = jax.random.PRNGKey(SEED)
WIDTH_x_HEIGHT = 512 # 512 default, 256 because vram

# 4.  Prepare prompt IDs and run
prompt_ids = pipe.prepare_inputs([PROMPT] * NUM_PARTICLES)
print("Generating images …")
out = pipe(
    prompt_ids,
    params,
    prng,
    num_inference_steps=TIME_STEPS,
    height=WIDTH_x_HEIGHT,
    width=WIDTH_x_HEIGHT,
    jit=False,                    # FK-Steering needs the Python loop (so we cant use JIT here)
    fkd_args=fkd_cfg,
    prompts=[PROMPT] * NUM_PARTICLES,
)

# 5.  Save / show results
images = pipe.numpy_to_pil(np.asarray(out.images))
# Full experiment config
experiment_config = {
    "seed": SEED,
    "prompt": PROMPT,
    "pipeline": "CompVis/stable-diffusion-v1-4",
    "revision": "bf16",
    "dtype": "float16",
    "jit": False,
    "width x height": WIDTH_x_HEIGHT,
    **fkd_cfg
}
# Save config
output_dir = create_output_dir(SEED, experiment_config)
with open(output_dir / "config.json", "w") as f:
    json.dump(experiment_config, f, indent=2)
# Save images
save_image_grid(images, output_dir / "fkd_results.png")