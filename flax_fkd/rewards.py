import json
import jax
import jax.numpy as jnp
from .flax_ImageReward import rm_load
from .flax_CLIP import FlaxCLIPScore
from .GenEval import GenEval

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
        model = FlaxCLIPScore(dtype=dtype)
        dummy_image_input = jnp.zeros((1, 3, 512, 512), dtype=dtype)
        params = model.init(jax.random.PRNGKey(0), "", dummy_image_input)
        REWARDS_DICT["Clip-Score"] = (model, params)

    model, params = REWARDS_DICT["Clip-Score"]
    clip_result = [
        model.apply(params, images=images[i], prompts=prompt)
        for i, prompt in enumerate(prompts)
    ]
    return clip_result

# Compute LLM-grading
def do_llm_grading(*, images, prompts, metric_to_chase="overall_score", dtype=jnp.float32):
    raise NotImplementedError("LLM grading reward has not been implemented yet")

def do_geneval_score(*, image_names, images, metadata):
    # If metadata path provided, read file
    if isinstance(metadata, str):
        if metadata.endswith(".json"):
            with open(metadata, "r") as f:
                metadata = json.load(f)
        else:
            assert metadata.endswith(".jsonl")
            with open(metadata, "r") as f:
                metadata = [json.loads(line) for line in f]
    # Make list
    if not isinstance(metadata, list):
        metadata = [metadata]

    # Otherwise, assume correct data format
    genEval = GenEval()
    score = genEval.evaluate_many(image_names=image_names, images=images, metadatas=metadata)

    return score
    