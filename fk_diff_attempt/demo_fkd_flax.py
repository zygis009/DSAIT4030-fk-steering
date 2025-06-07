# demo_fkd_flax.py  ───────────────────────────────────────────────────────────
import os
import jax
import numpy as np
from diffusers import FlaxDDIMScheduler
from fkd_pipeline_sd14_flax import FKDFlaxStableDiffusion
from plot_helper import save_image_grid
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
    potential_type     = "diff",        # or "diff", "add"
    lmbda              = 2.0,          # paper default
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
SEED = 0
prng = jax.random.PRNGKey(SEED)


# 4.  Prepare prompt IDs and run
prompt_ids = pipe.prepare_inputs([PROMPT] * NUM_PARTICLES)

print("Generating …")
out = pipe(
    prompt_ids,
    params,
    prng,
    num_inference_steps=TIME_STEPS,
    jit=False,                    # FK-Steering needs the Python loop (so we cant use JIT here)
    fkd_args=fkd_cfg,
    prompts=[PROMPT] * NUM_PARTICLES,
)

# 5.  Save / show results
images = pipe.numpy_to_pil(np.asarray(out.images))
os.makedirs("outputs", exist_ok=True)
for i, img in enumerate(images):
    fname = f"outputs/fkd_{i}.png"
    img.save(fname)
    print("Saved:", fname)
    
# 6.  Composite image of all final particles
save_image_grid(images, "outputs/fkd_grid.png")