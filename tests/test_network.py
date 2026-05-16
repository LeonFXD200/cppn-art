"""Tests for cppn_art.network"""

import numpy as np
import pytest
import sys

from cppn_art.network import CPPN, ACTIVATIONS, CPPNLayer, _HIDDEN_ACT_NAMES


class TestActivations:
    def test_all_activations_run(self):
        x = np.linspace(-3, 3, 50, dtype=np.float32)
        for name, fn in ACTIVATIONS.items():
            out = fn(x)
            assert out.shape == x.shape, f"{name} changed shape"
            assert np.all(np.isfinite(out)), f"{name} produced non-finite values"

    def test_sigmoid_range(self):
        x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32)
        s = ACTIVATIONS["sigmoid"](x)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_gauss_range(self):
        x = np.linspace(-10, 10, 100, dtype=np.float32)
        g = ACTIVATIONS["gauss"](x)
        assert np.all(g >= 0) and np.all(g <= 1)

    def test_hidden_acts_excludes_sigmoid(self):
        assert "sigmoid" not in _HIDDEN_ACT_NAMES, (
            "sigmoid must not be a random hidden-layer option; "
            "it collapses signal to (0,1) and can cause flat 0.5 output"
        )


class TestCPPNLayer:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        layer = CPPNLayer(4, 8, "tanh", rng)
        x = np.random.randn(16, 4).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (16, 8)

    def test_repr(self):
        rng = np.random.default_rng(0)
        layer = CPPNLayer(4, 8, "sin", rng)
        assert "4->8" in repr(layer)
        assert "sin" in repr(layer)


class TestCPPN:
    def test_default_init(self):
        cppn = CPPN(seed=42)
        assert cppn.seed == 42
        assert len(cppn.layers) >= 1
        assert cppn.z.shape == (8,)

    def test_forward_shape_2d(self):
        cppn = CPPN(seed=1)
        X = np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)
        Y = np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)
        out = cppn.forward(X, Y)
        assert out.shape == (4, 4, 3)

    def test_forward_shape_1d(self):
        cppn = CPPN(seed=2)
        x = np.linspace(-1, 1, 100, dtype=np.float32)
        y = np.zeros(100, dtype=np.float32)
        out = cppn.forward(x, y)
        assert out.shape == (100, 3)

    def test_output_range(self):
        cppn = CPPN(seed=3)
        X = np.linspace(-1, 1, 32, dtype=np.float32).reshape(4, 8)
        Y = np.zeros((4, 8), dtype=np.float32)
        out = cppn.forward(X, Y)
        assert np.all(out >= 0.0), "Values below 0"
        assert np.all(out <= 1.0), "Values above 1"

    def test_deterministic(self):
        cppn1 = CPPN(seed=99)
        cppn2 = CPPN(seed=99)
        X = np.linspace(-1, 1, 8, dtype=np.float32)
        Y = np.zeros(8, dtype=np.float32)
        np.testing.assert_array_equal(cppn1.forward(X, Y), cppn2.forward(X, Y))

    def test_different_seeds_differ(self):
        """Different seeds should produce visually distinct images."""
        X = np.linspace(-1, 1, 32, dtype=np.float32)
        Y = np.zeros(32, dtype=np.float32)
        pairs = [(42, 137), (1000, 2000), (314, 271)]
        for s1, s2 in pairs:
            out1 = CPPN(seed=s1).forward(X, Y)
            out2 = CPPN(seed=s2).forward(X, Y)
            if not np.allclose(out1, out2, atol=1e-3):
                return
        pytest.fail("All seed pairs produced identical output")

    def test_output_not_flat(self):
        """Each seed should produce images with visible variation across pixels."""
        X = np.linspace(-1, 1, 64, dtype=np.float32)
        Y = np.linspace(-1, 1, 64, dtype=np.float32)
        for seed in [42, 137, 1000, 2718]:
            out = CPPN(seed=seed).forward(X, Y)
            std = out.std()
            assert std > 1e-3, (
                f"seed={seed} produced near-flat output (std={std:.6f}). "
                "Network signal may be collapsing."
            )

    def test_time_varies_output(self):
        """t=0.0 and t=0.5 should produce different images (the animation moves)."""
        X = np.linspace(-1, 1, 32, dtype=np.float32)
        Y = np.zeros(32, dtype=np.float32)
        for seed in [42, 137, 1000]:
            out0 = CPPN(seed=seed).forward(X, Y, t=0.0)
            out5 = CPPN(seed=seed).forward(X, Y, t=0.5)
            if not np.allclose(out0, out5, atol=1e-3):
                return
        pytest.fail("No seed showed time-varying output")

    def test_custom_architecture(self):
        cppn = CPPN(seed=7, hidden_sizes=[8, 16, 8], activations=["sin", "tanh", "cos"])
        assert cppn.hidden_sizes == [8, 16, 8]
        assert cppn.activations  == ["sin", "tanh", "cos"]

    def test_activation_mismatch_raises(self):
        with pytest.raises(ValueError):
            CPPN(seed=0, hidden_sizes=[8, 16], activations=["sin"])

    def test_describe_string(self):
        cppn = CPPN(seed=42)
        desc = cppn.describe()
        assert "seed=42" in desc
        assert "layers=" in desc

    def test_no_nan_outputs(self):
        for seed in range(20):
            cppn = CPPN(seed=seed)
            X = np.random.default_rng(seed).uniform(-2, 2, (8, 8)).astype(np.float32)
            Y = np.random.default_rng(seed + 100).uniform(-2, 2, (8, 8)).astype(np.float32)
            out = cppn.forward(X, Y, t=0.3)
            assert np.all(np.isfinite(out)), f"NaN/Inf detected for seed={seed}"

    def test_all_presets_produce_valid_output(self):
        """Every built-in preset should render without error and produce valid pixels."""
        from cppn_art.presets import PRESETS
        from cppn_art.renderer import render_frame
        for name, data in PRESETS.items():
            cppn = CPPN(seed=data["seed"], hidden_sizes=data["hidden_sizes"])
            frame = render_frame(cppn, width=32, height=32)
            assert frame.shape == (32, 32, 3), f"Preset {name}: wrong shape"
            assert frame.dtype == np.uint8,    f"Preset {name}: wrong dtype"
            assert np.all(frame >= 0) and np.all(frame <= 255), f"Preset {name}: out of range"
