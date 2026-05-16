#!/usr/bin/env python3
"""
Example: Generate a gallery of 16 random CPPN patterns.

Run:
    python examples/generate_gallery.py
"""

import sys
sys.path.insert(0, ".")

from cppn_art import CPPN
from cppn_art.renderer import render_image, render_gallery, render_gif
from cppn_art.presets import PRESETS

def example_single():
    print("Generating single image (seed=42)...")
    img = render_image(CPPN(seed=42), width=512, height=512)
    img.save("example_single.png")
    print("  Saved: example_single.png")

def example_presets():
    print("\nGenerating all presets...")
    for name, data in PRESETS.items():
        cppn = CPPN(seed=data["seed"], hidden_sizes=data["hidden_sizes"])
        img = render_image(cppn, width=256, height=256)
        filename = f"preset_{name}.png"
        img.save(filename)
        print(f"  {data['emoji']}  {name:12}  →  {filename}")

def example_gallery():
    print("\nGenerating 16-seed gallery...")
    render_gallery(
        seeds=list(range(1, 17)),
        output_path="gallery_seeds_1_16.png",
        cols=4,
        cell_size=256,
    )
    print("  Saved: gallery_seeds_1_16.png")

def example_animation():
    print("\nGenerating animated GIF (seed=137, 60 frames)...")
    render_gif(
        CPPN(seed=137),
        output_path="animation_seed137.gif",
        width=256,
        height=256,
        frames=60,
        fps=20,
    )
    print("  Saved: animation_seed137.gif")

if __name__ == "__main__":
    example_single()
    example_presets()
    example_gallery()
    # Uncomment to generate GIF (takes ~30s):
    # example_animation()
    print("\nDone! ✓")
