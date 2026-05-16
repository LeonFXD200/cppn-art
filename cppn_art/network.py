from __future__ import annotations

import numpy as np
from typing import List, Optional


def _safe_sigmoid(x):
    pos = x >= 0
    result = np.empty_like(x, dtype=np.float32)
    result[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result

def _sinc(x):
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(x == 0.0, 1.0, np.sin(np.pi * x) / (np.pi * x))
    return out.astype(np.float32)

def _softplus(x):
    return np.log1p(np.exp(np.clip(x, -500, 20))).astype(np.float32)

ACTIVATIONS = {
    'sin':      lambda x: np.sin(x).astype(np.float32),
    'cos':      lambda x: np.cos(x).astype(np.float32),
    'tanh':     lambda x: np.tanh(x).astype(np.float32),
    'relu':     lambda x: np.maximum(0.0, x).astype(np.float32),
    'abs':      lambda x: np.abs(x).astype(np.float32),
    'gauss':    lambda x: np.exp(-(x ** 2)).astype(np.float32),
    'sinc':     _sinc,
    'softplus': _softplus,
    'elu':      lambda x: np.where(x >= 0, x, np.expm1(np.clip(x, -500, 0))).astype(np.float32),
    'sigmoid':  _safe_sigmoid,
}

_HIDDEN_ACT_NAMES = ['sin', 'cos', 'tanh', 'relu', 'abs', 'gauss', 'sinc', 'softplus', 'elu']


class CPPNLayer:
    def __init__(self, in_size, out_size, activation, rng):
        self.activation_name = activation
        self.fn = ACTIVATIONS[activation]
        scale = np.sqrt(2.0 / in_size)
        self.W = rng.normal(0.0, scale, (in_size, out_size)).astype(np.float32)
        self.b = rng.normal(0.0, 0.1, (out_size,)).astype(np.float32)

    def forward(self, x):
        return self.fn(x @ self.W + self.b)

    def __repr__(self):
        return f'CPPNLayer({self.W.shape[0]}->{self.W.shape[1]}, act={self.activation_name})'


class CPPN:
    """
    Compositional Pattern Producing Network.

    Maps pixel coordinates (x, y) + time t to RGB colors.
    Every seed produces a completely different, infinitely detailed pattern.

    Parameters
    ----------
    seed : int
        Random seed -- the DNA of your artwork.
    hidden_sizes : list[int], optional
        Width of each hidden layer.
    activations : list[str], optional
        Activation function per hidden layer.
    z_dim : int
        Size of the per-seed latent vector.

    Examples
    --------
    >>> from cppn_art import CPPN
    >>> from cppn_art.renderer import render_image
    >>> img = render_image(CPPN(seed=42))
    >>> img.save('cosmos.png')
    """

    def __init__(self, seed=42, hidden_sizes=None, activations=None, z_dim=8):
        self.seed = seed
        self.z_dim = z_dim
        rng = np.random.default_rng(seed)
        self.z = rng.normal(0.0, 1.0, z_dim).astype(np.float32)

        if hidden_sizes is None:
            depth = int(rng.integers(3, 7))
            hidden_sizes = [int(rng.choice([16, 32, 64, 128])) for _ in range(depth)]

        if activations is None:
            activations = [
                _HIDDEN_ACT_NAMES[rng.integers(0, len(_HIDDEN_ACT_NAMES))]
                for _ in range(len(hidden_sizes))
            ]

        if len(activations) != len(hidden_sizes):
            raise ValueError(
                f'len(activations)={len(activations)} must equal '
                f'len(hidden_sizes)={len(hidden_sizes)}'
            )

        self.hidden_sizes = list(hidden_sizes)
        self.activations = list(activations)

        in_size = 5 + z_dim
        self.layers = []
        prev = in_size
        for size, act in zip(hidden_sizes, activations):
            self.layers.append(CPPNLayer(prev, size, act, rng))
            prev = size

        self.out_layer = CPPNLayer(prev, 3, 'sigmoid', rng)

    def forward(self, x, y, t=0.0):
        """
        Evaluate the network at a grid of (x, y) coordinates.

        Parameters
        ----------
        x, y : ndarray, shape (H, W) or (N,)
            Coordinates in [-1, 1].
        t : float
            Time in [0, 1] for animation.

        Returns
        -------
        ndarray, shape (*x.shape, 3)
            RGB values in [0, 1].
        """
        shape = x.shape
        xf = x.ravel().astype(np.float32)
        yf = y.ravel().astype(np.float32)
        n = len(xf)

        r = np.sqrt(xf ** 2 + yf ** 2)
        sin_t = np.full(n, np.sin(2 * np.pi * t), dtype=np.float32)
        cos_t = np.full(n, np.cos(2 * np.pi * t), dtype=np.float32)
        z_rep = np.tile(self.z, (n, 1))

        h = np.column_stack([xf, yf, r, sin_t, cos_t, z_rep])
        for layer in self.layers:
            h = layer.forward(h)
        rgb = self.out_layer.forward(h)
        return rgb.reshape(*shape, 3)

    def describe(self):
        acts = ' -> '.join(self.activations)
        sizes = ' -> '.join(str(s) for s in self.hidden_sizes)
        return f'CPPN(seed={self.seed} | layers=[{sizes}] | activations=[{acts}])'

    def __repr__(self):
        return self.describe()
