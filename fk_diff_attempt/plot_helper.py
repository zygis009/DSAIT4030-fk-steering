# ── helper: build a collage of PIL images ──────────────────────────────
from math import ceil, sqrt
from PIL import Image
import os

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