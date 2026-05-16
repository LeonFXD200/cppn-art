from __future__ import annotations
import time
from typing import List, Sequence, Tuple
import numpy as np


def make_grid(width, height, scale=1.0):
    xs = np.linspace(-scale, scale, width, dtype=np.float32)
    ys = np.linspace(-scale, scale, height, dtype=np.float32)
    return np.meshgrid(xs, ys)


def render_frame(cppn, width=512, height=512, t=0.0, scale=1.0):
    """Render a single frame as an (H, W, 3) uint8 numpy array."""
    X, Y = make_grid(width, height, scale)
    rgb = cppn.forward(X, Y, t)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def render_image(cppn, width=512, height=512, t=0.0, scale=1.0):
    """Render and return a PIL Image."""
    from PIL import Image
    arr = render_frame(cppn, width, height, t, scale)
    return Image.fromarray(arr, 'RGB')


def render_gif(cppn, output_path, width=256, height=256, frames=60, fps=20, scale=1.0, verbose=True):
    """
    Render a seamlessly looping animated GIF.

    The animation is periodic: t goes from 0 to 1 (exclusive), so frame N
    smoothly connects back to frame 0.
    """
    from PIL import Image
    t0 = time.time()
    pil_frames = []
    for i in range(frames):
        t = i / frames
        if verbose:
            pct = (i + 1) / frames * 100
            bar = chr(9608) * int(pct // 5) + chr(9617) * (20 - int(pct // 5))
            print(f'\r  [{bar}] {pct:5.1f}%  frame {i+1}/{frames}', end='', flush=True)
        arr = render_frame(cppn, width, height, t, scale)
        pil_frames.append(Image.fromarray(arr, 'RGB'))
    if verbose:
        print(f'\n  Rendered {frames} frames in {time.time()-t0:.1f}s')
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=duration_ms, optimize=False,
    )


def render_gallery(seeds, output_path, cols=4, cell_size=256, t=0.0, verbose=True, **cppn_kwargs):
    """Render a grid of images (one per seed) and save as a single PNG."""
    from PIL import Image
    from .network import CPPN
    seeds = list(seeds)
    rows = (len(seeds) + cols - 1) // cols
    W, H = cols * cell_size, rows * cell_size
    canvas = Image.new('RGB', (W, H), (8, 8, 12))
    for idx, seed in enumerate(seeds):
        col, row = idx % cols, idx // cols
        if verbose:
            print(f'  [{idx+1:>{len(str(len(seeds)))}}/{len(seeds)}]  seed={seed}')
        cppn = CPPN(seed=seed, **cppn_kwargs)
        img = render_image(cppn, cell_size, cell_size, t=t)
        canvas.paste(img, (col * cell_size, row * cell_size))
    canvas.save(output_path)
    if verbose:
        print(f'  Saved {len(seeds)}-image gallery -> {output_path}')
