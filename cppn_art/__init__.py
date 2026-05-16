from .network import CPPN, ACTIVATIONS, CPPNLayer
from .renderer import render_frame, render_image, render_gif, render_gallery
from . import presets

__version__ = "0.1.0"
__author__ = "LeonFXD200"
__license__ = "MIT"

__all__ = [
    "CPPN",
    "CPPNLayer",
    "ACTIVATIONS",
    "render_frame",
    "render_image",
    "render_gif",
    "render_gallery",
    "presets",
]
