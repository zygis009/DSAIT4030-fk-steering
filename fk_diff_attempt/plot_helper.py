# ── helper: build a collage of PIL images ──────────────────────────────
from math import ceil, sqrt
from PIL import Image
import os
import hashlib
import json
import hashlib
from pathlib import Path
from datetime import datetime

def save_image_grid(pil_list, fname, padding=8, bg=(255, 255, 255)):
    """
    Args
    ----
    pil_list : list[PIL.Image]   – images of identical size
    fname    : str               – output path (PNG/JPEG)
    padding  : int               – pixels between tiles
    bg       : tuple[int,int,int]– RGB background colour
    """
    if len(pil_list) == 0:
        raise ValueError("No images to tile.")

    # determine grid shape: as square as possible
    n   = len(pil_list)
    cols = int(ceil(sqrt(n)))
    rows = int(ceil(n / cols))

    w, h = pil_list[0].size
    W = cols * w + (cols - 1) * padding
    H = rows * h + (rows - 1) * padding

    grid = Image.new("RGB", (W, H), bg)

    for idx, im in enumerate(pil_list):
        r, c = divmod(idx, cols)
        x = c * (w + padding)
        y = r * (h + padding)
        grid.paste(im, (x, y))

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    grid.save(fname)
    print("Saved grid:", fname)
    
def config_hash(config: dict, length=8):
    """Generate a short hash for a config dictionary."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:length]

def create_output_dir(seed, config, base="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = config_hash(config)
    dir_name = f"run_{timestamp}_{cfg_hash}_seed{seed}"
    output_path = Path(base) / dir_name
    output_path.mkdir(parents=True, exist_ok=False)
    return output_path