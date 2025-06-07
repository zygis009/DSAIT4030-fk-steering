import jax.numpy as jnp
from .flax_ImageReward import rm_load
from .flax_CLIP import FlaxCLIPScore

# Stores the reward models
REWARDS_DICT = {
    "ImageReward": None,
    "Clip-Score": None,
    "LLMGrader": None,
}

# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score", dtype=jnp.float32):
    if reward_name != "LLMGrader":
        print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    if reward_name == "ImageReward":
        return do_image_reward(images=images, prompts=prompts, dtype=dtype)
    
    elif reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts, dtype=dtype)
    
    elif reward_name == "LLMGrader":
        return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase, dtype=dtype)
    
    else:
        raise ValueError(f"Unknown metric: {reward_name}")

# Compute ImageReward
def do_image_reward(*, images, prompts, dtype=jnp.float32):
    global REWARDS_DICT
    if REWARDS_DICT["ImageReward"] is None:
        REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

    image_reward_result = REWARDS_DICT["ImageReward"](images, prompts)

    return image_reward_result

# Compute CLIP-Score
def do_clip_score(*, images, prompts, dtype=jnp.float32):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = FlaxCLIPScore(dtype=dtype)
    clip_result = [
        REWARDS_DICT["Clip-Score"](prompt, images[i])
        for i, prompt in enumerate(prompts)
    ]
    return clip_result

# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score", dtype=jnp.float32):
    raise NotImplementedError("LLM grading reward has not been implemented yet")
