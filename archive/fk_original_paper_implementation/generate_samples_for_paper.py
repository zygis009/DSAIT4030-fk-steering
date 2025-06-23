import sys
sys.path.append('fkd_diffusers')

import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import numpy as np
from copy import deepcopy

import torch

from fks_utils import get_model, do_eval


args = dict(
    output_dir="samples_for_paper",
    eta=1.0,
    guidance_reward_fn="ImageReward",
    metrics_to_compute="ImageReward",
    seed=42,
)

args = argparse.Namespace(**args)
print(args)

# cache metric fns
do_eval(
    prompt=["test"],
    images=[Image.new("RGB", (224, 224))],
    metrics_to_compute=["ImageReward"],
)


# seed everything
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def generate_config():
    base_fkd_args = dict(
        lmbda=2.0,
        use_smc=True,
        adaptive_resampling=True,
        resample_frequency=20,
        resampling_t_start=20,
        resampling_t_end=80,
        guidance_reward_fn="ImageReward",
        metric_to_chase=None, # should be specified when using "LLMGrader".
    )

    arr_fkd_args = []

    for time_steps in [100]:
        for lmbda in [2.0]:
            for num_particles in [4]:
                base_fkd_args["time_steps"] = time_steps
                base_fkd_args["lmbda"] = lmbda
                base_fkd_args["num_particles"] = num_particles
                arr_fkd_args.append(base_fkd_args.copy())

    return arr_fkd_args


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_and_save_image(images, image_fpath, num_particles):
    if num_particles > 1:
        fig, ax = plt.subplots(
            1, num_particles, figsize=(num_particles * 5, 5), dpi=200
        )

        for i, image in enumerate(images):
            ax[i].imshow(image)
            ax[i].axis("off")

        fig.tight_layout()
        plt.savefig(image_fpath)
        plt.show()
        plt.close()
    else:
        plt.imshow(images[0])
        plt.axis("off")
        plt.savefig(image_fpath)
        plt.show()
        plt.close()


def generate_samples(fkd_args, pipeline, prompt_data):
    for prompt_idx, item in enumerate(prompt_data):
        prompt = [item["prompt"]] * fkd_args["num_particles"]
        prompt_ = item["prompt"].replace(" ", "_")
        lmbda_ = fkd_args["lmbda"]
        num_particles = fkd_args["num_particles"]
        time_steps_ = fkd_args["time_steps"]

        image_fpath = os.path.join(images_path, prompt_)
        os.makedirs(image_fpath, exist_ok=True)

        file_name = f"seed_{prompt_idx}_lmbda_{lmbda_}_time_steps_{time_steps_}_num_particles_{num_particles}"

        max_fname = os.path.join(image_fpath, f"{file_name}_max.png")
        diff_fname = os.path.join(image_fpath, f"{file_name}_diff.png")
        base_fname = os.path.join(image_fpath, f"{file_name}_base.png")

        os.makedirs(image_fpath, exist_ok=True)
        print(f"Generating samples for {image_fpath}")

        seed_ = 0 + prompt_idx
        seed_everything(seed_)

        fkd_max_args = deepcopy(fkd_args)
        fkd_max_args["potential_type"] = "max"
        print(f"Generating samples for {fkd_max_args}")
        images_fkd_max = pipeline(
            prompt,
            num_inference_steps=fkd_args["time_steps"],
            eta=args.eta,
            fkd_args=fkd_max_args,
        )[0]
        results = do_eval(
            prompt=prompt,
            images=images_fkd_max,
            metrics_to_compute=["ImageReward"],
        )
        # sort images by reward
        guidance_reward = np.array(results["ImageReward"]["result"])
        sorted_idx = np.argsort(guidance_reward)[::-1]
        images_fkd_max = [images_fkd_max[i] for i in sorted_idx]

        generate_and_save_image(images_fkd_max, max_fname, num_particles)

        seed_everything(0 + prompt_idx)

        fkd_diff_args = deepcopy(fkd_args)
        fkd_diff_args["potential_type"] = "diff"
        print(f"Generating samples for {fkd_diff_args}")
        images_fkd_diff = pipeline(
            prompt,
            num_inference_steps=fkd_args["time_steps"],
            eta=args.eta,
            fkd_args=fkd_diff_args,
        )[0]
        generate_and_save_image(images_fkd_diff, diff_fname, num_particles)

        seed_everything(0 + prompt_idx)
        base_args = deepcopy(fkd_args)
        base_args["use_smc"] = False
        print(f"Generating samples for {base_args}")
        images_base = pipeline(
            [item["prompt"]] * 4,
            num_inference_steps=fkd_args["time_steps"],
            eta=args.eta,
            fkd_args=base_args,
        )[0]
        generate_and_save_image(images_base, base_fname, num_particles=4)


prompt_data = [
    {"prompt": "a photo of a brown knife and a blue donut"},
    {"prompt": "a photo of a blue clock and a white cup"},
    {"prompt": "a photo of an orange cow and a purple sandwich"},
    {"prompt": "a photo of a yellow bird and a black motorcycle"},
    {"prompt": "a photo of a green tennis racket and a black dog"},
    {"prompt": "a green stop sign in a red field"},
]


for model_name in [
    "stable-diffusion-2-1",
    "stable-diffusion-xl",
]:
    # load model
    pipeline = get_model(model_name)

    # set output directory
    arr_fkd_args = generate_config()
    output_dir = os.path.join(args.output_dir)
    output_dir += f"_{args.metrics_to_compute}" 
    if arr_fkd_args[0]["metric_to_chase"]:
        output_dir += f'_{arr_fkd_args[0]["metric_to_chase"]}'
    os.makedirs(output_dir, exist_ok=True)

    images_path = output_dir + f"/{model_name}"
    os.makedirs(images_path, exist_ok=True)

    pipeline = pipeline.to("cuda")

    # generate samples
    for fkd_args in arr_fkd_args:
        print(fkd_args)
        for prompt in prompt_data:
            generate_samples(fkd_args, pipeline, [prompt])
