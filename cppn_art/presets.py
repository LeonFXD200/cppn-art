from __future__ import annotations
from typing import Dict, Any

PRESETS: Dict[str, Dict[str, Any]] = {
    'cosmos': {
        'seed': 137,
        'hidden_sizes': [32, 64, 64, 32],
        'description': 'Deep space nebula -- swirling indigos and violet clouds',
        'emoji': '\U0001f30c',
    },
    'coral': {
        'seed': 2718,
        'hidden_sizes': [16, 64, 64, 64, 16],
        'description': 'Organic coral branching -- soft pinks and warm oranges',
        'emoji': '\U0001fab8',
    },
    'aurora': {
        'seed': 42,
        'hidden_sizes': [32, 32, 64, 32, 32],
        'description': 'Northern lights -- flowing greens and electric teals',
        'emoji': '\U0001f30c',
    },
    'mandala': {
        'seed': 9999,
        'hidden_sizes': [64, 128, 64],
        'description': 'Radially symmetric mandala -- jewel-like symmetry',
        'emoji': '\U0001f52e',
    },
    'lava': {
        'seed': 1337,
        'hidden_sizes': [32, 64, 32],
        'description': 'Hot lava flow -- deep reds bleeding into molten gold',
        'emoji': '\U0001f30b',
    },
    'tide': {
        'seed': 314159,
        'hidden_sizes': [16, 32, 64, 32, 16],
        'description': 'Tidal foam -- cerulean and sea-glass ripples',
        'emoji': '\U0001f30a',
    },
    'electric': {
        'seed': 777,
        'hidden_sizes': [32, 64, 128, 64, 32],
        'description': 'Lightning discharge -- white-hot filaments on midnight blue',
        'emoji': '\u26a1',
    },
    'silk': {
        'seed': 2023,
        'hidden_sizes': [64, 64, 64, 64],
        'description': 'Iridescent silk -- smooth, shifting colour fields',
        'emoji': '\U0001f308',
    },
    'jungle': {
        'seed': 8675309,
        'hidden_sizes': [16, 32, 32, 64, 32, 16],
        'description': 'Tropical canopy -- layered greens with golden shafts',
        'emoji': '\U0001f33f',
    },
    'rorschach': {
        'seed': 101010,
        'hidden_sizes': [32, 128, 128, 32],
        'description': 'Symmetric inkblot -- high-contrast monochrome forms',
        'emoji': '\U0001f5a4',
    },
}


def get(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]
