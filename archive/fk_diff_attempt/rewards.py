"""
Thin wrappers around the reward models used for steering.
ImageReward is mandatory; CLIP-Score & HPS-v2 are optional.
All heavy lifting stays in PyTorch/CPU so JAX code is unaffected.
"""
import torch
from PIL import Image
from typing import List
from image_reward_utils import rm_load     # pip install ImageReward==1.0
# from llm_grading         import LLMGrader  # optional, only if installed
import hpsv2
import clip
import torch.nn.functional as F


# keep reward model instances in a dict to avoid re-loading
CACHE = dict()

# ---------------------------------------------------------------------
def _clip_score(prompt:str, pil_image:Image.Image):
    if "clip" not in CACHE:
        model, preprocess = clip.load("ViT-L/14", device="cpu")
        model.eval(); CACHE["clip"] = (model, preprocess)
    model, preprocess = CACHE["clip"]
    with torch.no_grad():
        txt = clip.tokenize(prompt).to("cpu")
        img = preprocess(pil_image).unsqueeze(0)
        t_f = F.normalize(model.encode_text(txt))
        i_f = F.normalize(model.encode_image(img))
        return (t_f * i_f).sum().item()


def get_reward_function(name:str, *, images:List[Image.Image],
                        prompts:List[str], metric_to_chase=None):
    if name == "ImageReward":
        if "ir" not in CACHE:
            CACHE["ir"] = rm_load("ImageReward-v1.0", device="cpu")
        return CACHE["ir"].score_batched(prompts, images)

    if name == "Clip-Score":
        return [_clip_score(p, im) for p, im in zip(prompts, images)]

    if name == "HumanPreference":
        return [float(hpsv2.score(im, prmpt, hps_version="v2.1")[0])
                for im, prmpt in zip(images, prompts)]

    # if name == "LLMGrader":
    #     if "llm" not in CACHE:
    #         CACHE["llm"] = LLMGrader()
    #     return [CACHE["llm"].score(images=im, prompts=prmpt,
    #                                metric_to_chase=metric_to_chase)
    #             for im, prmpt in zip(images, prompts)]

    raise ValueError(f"Unknown reward: {name}")