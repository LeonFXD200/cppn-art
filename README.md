<div align="center">

# 🎨 cppn-art

**Generate infinite unique fractal artwork using tiny neural networks.**

[![CI](https://github.com/LeonFXD200/cppn-art/actions/workflows/ci.yml/badge.svg)](https://github.com/LeonFXD200/cppn-art/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-only-green.svg)](https://numpy.org)

*No GPU. No pretrained weights. No API keys. Just maths and imagination.*

</div>

---

## What is this?

**CPPN-Art** uses [Compositional Pattern Producing Networks](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) — a clever trick where a tiny neural network **is** the artwork.

Instead of training a network on images, we ask: *"what if a neural network's job was simply to answer — given coordinates (x, y), what colour should this pixel be?"*

The network takes a pixel's position and outputs an RGB colour. Evaluated across a whole image grid, this produces **infinitely detailed, seamlessly tileable, endlessly animatable patterns** — all from a single integer seed.

```
(x, y, r, t)  ──►  [hidden layers with exotic activations]  ──►  (R, G, B)
                         sin  cos  tanh  gauss  sinc  elu ...
```

Every seed = a different network = a completely different universe of patterns.

---

## Gallery

| 🌌 Cosmos (seed 137) | 🪸 Coral (seed 2718) | ⚡ Electric (seed 777) | 🌊 Tide (seed 314159) |
|:---:|:---:|:---:|:---:|
| Deep space nebula | Organic branching | Lightning discharge | Tidal foam ripples |

| 🌋 Lava (seed 1337) | 🌈 Silk (seed 2023) | 🔮 Mandala (seed 9999) | 🌿 Jungle (seed 8675309) |
|:---:|:---:|:---:|:---:|
| Molten reds & gold | Iridescent colour fields | Jewel-like symmetry | Tropical canopy |

> **Try any of these with** `cppn-art generate --preset <name>` or explore the infinite space with `--seed <any integer>`.

---

## Installation

```bash
pip install cppn-art
```

Or from source:

```bash
git clone https://github.com/LeonFXD200/cppn-art.git
cd cppn-art
pip install -e ".[dev]"
```

**Requirements:** Python 3.9+, NumPy, Pillow. That's it.

---

## Quick start

### Python API

```python
from cppn_art import CPPN
from cppn_art.renderer import render_image, render_gif, render_gallery

# Single image
img = render_image(CPPN(seed=42))
img.save("cosmos.png")

# High resolution
img = render_image(CPPN(seed=137), width=2048, height=2048)
img.save("cosmos_4k.png")

# Animated GIF (seamlessly looping)
render_gif(CPPN(seed=42), "aurora.gif", frames=60, fps=20)

# Gallery: 16 seeds in one image
render_gallery(seeds=range(1, 17), output_path="gallery.png", cols=4, cell_size=256)

# Custom architecture
cppn = CPPN(
    seed=999,
    hidden_sizes=[32, 64, 128, 64, 32],
    activations=["sin", "tanh", "cos", "gauss", "sin"],
)
render_image(cppn, width=1024, height=1024).save("custom.png")
```

### Command line

```bash
cppn-art generate --seed 42
cppn-art generate --preset cosmos --width 1024 --height 1024
cppn-art animate  --preset aurora --frames 90 --fps 30
cppn-art gallery  --start 1 --count 16 --cols 4
cppn-art presets
cppn-art info --seed 42
```

---

## How it works

A CPPN maps pixel coordinates (x, y) + time t to RGB. Inputs are:
- **x, y** — pixel coordinates in [-1, 1]
- **r** — distance from centre
- **sin(2πt), cos(2πt)** — periodic time encoding (enables seamless loops)
- **z₁…z₈** — per-seed latent personality vector

With exotic activations (sin, gauss, sinc, tanh...) composed through multiple layers, the network produces hierarchical frequency patterns that look strikingly like natural phenomena — nebulae, coral, aurora, lava.

The periodic time encoding means animations loop **seamlessly by construction**. No blending tricks. The maths just works.

---

## Activation functions

| Name | Formula | Character |
|------|---------|-----------| 
| `sin` | sin(x) | Waves, periodicity |
| `cos` | cos(x) | Shifted waves |
| `tanh` | tanh(x) | Smooth boundaries |
| `relu` | max(0, x) | Sharp edges |
| `abs` | |x| | Reflective folds |
| `gauss` | exp(-x²) | Radial blobs |
| `sinc` | sin(πx)/(πx) | Interference rings |
| `softplus` | log(1+eˣ) | Smooth relu |
| `elu` | x if x≥0 else eˣ-1 | Negative reach |

---

## Built-in presets

| Preset | Seed | Description |
|--------|------|-------------|
| `cosmos` | 137 | 🌌 Deep space nebula — swirling indigos and violet clouds |
| `coral` | 2718 | 🪸 Organic coral branching — soft pinks and warm oranges |
| `aurora` | 42 | 🌌 Northern lights — flowing greens and electric teals |
| `mandala` | 9999 | 🔮 Radially symmetric — jewel-like symmetry |
| `lava` | 1337 | 🌋 Hot lava flow — deep reds bleeding into molten gold |
| `tide` | 314159 | 🌊 Tidal foam — cerulean and sea-glass ripples |
| `electric` | 777 | ⚡ Lightning discharge — white-hot filaments on midnight |
| `silk` | 2023 | 🌈 Iridescent silk — smooth, shifting colour fields |
| `jungle` | 8675309 | 🌿 Tropical canopy — layered greens with golden shafts |
| `rorschach` | 101010 | 🖤 Symmetric inkblot — high-contrast monochrome forms |

---

## API reference

### `CPPN(seed, hidden_sizes, activations, z_dim)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` | `42` | Random seed — the DNA of your artwork |
| `hidden_sizes` | `list[int]` | random | Width of each hidden layer |
| `activations` | `list[str]` | random | Activation per hidden layer |
| `z_dim` | `int` | `8` | Latent vector size |

### `render_image(cppn, width, height, t, scale) → PIL.Image`
Render a single frame as a PIL Image.

### `render_gif(cppn, output_path, width, height, frames, fps, scale)`
Render a seamlessly looping animated GIF.

### `render_gallery(seeds, output_path, cols, cell_size, t)`
Render a grid of images (one per seed) into a single PNG.

### `render_frame(cppn, width, height, t, scale) → np.ndarray`
Low-level render to (H, W, 3) uint8 array.

---

## Contributing

```bash
git clone https://github.com/LeonFXD200/cppn-art.git
cd cppn-art
pip install -e ".[dev]"
pytest
```

Ideas welcome: new activation functions, new presets (open a PR!), matplotlib live preview, video export.

---

## Background

CPPNs were introduced by [Kenneth O. Stanley (2007)](https://link.springer.com/article/10.1007/s10710-007-9028-8) as a way to encode biological pattern formation. This project applies that idea purely as an art tool.

---

## License

MIT © LeonFXD200
