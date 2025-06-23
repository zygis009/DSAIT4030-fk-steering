# rewards_flax.py

import torch
import numpy as np
from PIL import Image
import hpsv2
import clip

import ImageReward as RM

# Global dictionary to cache loaded models, preventing re-loading on every call
REWARD_MODELS = {
    "ImageReward": None,
    "HumanPreference": None,
    "Clip-Score": None
}

class SimpleCLIPScore:
    """A simple wrapper for CLIP score calculation."""
    def __init__(self, device):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device, jit=False)
        self.model.logit_scale.requires_grad_(False)

    @torch.no_grad()
    def score(self, prompt, pil_image):
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = torch.nn.functional.normalize(self.model.encode_text(text))
        
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = torch.nn.functional.normalize(self.model.encode_image(image))
        
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)
        return rewards.cpu().numpy().item()

def load_reward_models(device='cuda'):
    """Loads all required PyTorch-based reward models into memory."""
    print("Loading ImageReward model...")
    if REWARD_MODELS["ImageReward"] is None:
        REWARD_MODELS["ImageReward"] = RM.load("ImageReward-v1.0", device=device)
    
    print("HPSv2 model (Human Preference) is ready.")
    REWARD_MODELS["HumanPreference"] = True 

    print("Loading CLIP-Score model...")
    if REWARD_MODELS["Clip-Score"] is None:
        REWARD_MODELS["Clip-Score"] = SimpleCLIPScore(device=device)

    print("All reward models loaded.")

def get_reward_scores(images_np: np.ndarray, prompts: list, reward_name: str) -> np.ndarray:
    """
    Computes rewards for a batch of images. This function is designed to be
    called from JAX via host_callback. It runs on the host CPU.
    """
    images_pil = [Image.fromarray(img) for img in images_np]
    scores = []
    
    if reward_name == "ImageReward":
        model = REWARD_MODELS["ImageReward"]
        scores = model.score_batched(prompts, images_pil)

    elif reward_name == "HumanPreference":
        scores = hpsv2.score(images_pil, prompts, hps_version="v2.1")
        scores = [float(s) for s in scores]

    elif reward_name == "Clip-Score":
        model = REWARD_MODELS["Clip-Score"]
        for i, img in enumerate(images_pil):
            prompt_idx = i % len(prompts)
            scores.append(model.score(prompts[prompt_idx], img))
            
    else:
        raise ValueError(f"Unknown reward_name: {reward_name}")

    return np.array(scores, dtype=np.float32)