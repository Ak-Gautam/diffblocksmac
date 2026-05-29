"""Comprehensive tests for dblocks library.

Tests cover:
- Core math (schedules, blocks, sampling)
- NN modules (attention, conditioning, transformer)
- Model construction and forward pass
- Training step execution
- Memory profiler
"""

from __future__ import annotations

import math
import unittest
from random import Random

import mlx.core as mx
import mlx.nn as nn

# Core
from dblocks.core.blocks import BlockSpec, make_block_specs, partition_layers
from dblocks.core.sampling import PrecondCoeffs, edm_weight, preconditioning, sample_training_sigma
from dblocks.core.schedules import LogNormalNoise, dblock_inference_sigmas, edm_sigmas, equi_probability_sigmas

# NN
from dblocks.nn.attention import MultiHeadAttention, _create_causal_mask
from dblocks.nn.conditioning import AdaLN, TimestepEmbedder, modulate
from dblocks.nn.feedforward import FeedForward, SwiGLU
from dblocks.nn.transformer import DBlockTransformerLayer, TransformerBlock

# Models
from dblocks.models.gpt import DBlockGPT, GPTConfig

# Memory
from dblocks.memory.profiler import MemoryProfiler, memory_summary


# ============================================================
# Core math tests
# ============================================================

class TestLogNormalNoise(unittest.TestCase):
    def test_default_construction(self):
        noise = LogNormalNoise()
        self.assertAlmostEqual(noise.sigma_min, 0.002)
        self.assertAlmostEqual(noise.sigma_max, 80.0)

    def test_cdf_is_monotonic(self):
        noise = LogNormalNoise()
        sigmas = [0.01, 0.1, 1.0, 10.0, 50.0]
        cdfs = [noise.cdf(s) for s in sigmas]
        for i in range(len(cdfs) - 1):
            self.assertLess(cdfs[i], cdfs[i + 1])

    def test_ppf_inverts_cdf(self):
        noise = LogNormalNoise()
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sigma = noise.ppf(p)
            self.assertTrue(math.isclose(noise.cdf(sigma), p, rel_tol=1e-10))

    def test_validation(self):
        with self.assertRaises(ValueError):
            LogNormalNoise(sigma_min=-1)
        with self.assertRaises(ValueError):
            LogNormalNoise(sigma_max=0.001)
        with self.assertRaises(ValueError):
            LogNormalNoise(p_std=0)


class TestEquiProbabilitySigmas(unittest.TestCase):
    def test_equal_mass(self):
        noise = LogNormalNoise()
        boundaries = equi_probability_sigmas(4, noise=noise)
        self.assertEqual(len(boundaries), 5)
        self.assertTrue(math.isclose(boundaries[0], noise.sigma_min, abs_tol=1e-14))
        self.assertTrue(math.isclose(boundaries[-1], noise.sigma_max, rel_tol=1e-11))

        masses = [
            noise.cdf(boundaries[i + 1]) - noise.cdf(boundaries[i])
            for i in range(4)
        ]
        self.assertLess(max(masses) - min(masses), 1e-14)

    def test_single_block(self):
        b = equi_probability_sigmas(1)
        self.assertEqual(len(b), 2)
        self.assertAlmostEqual(b[0], 0.002)
        self.assertAlmostEqual(b[1], 80.0)


class TestBlockSpecs(unittest.TestCase):
    def test_reverse_noise_order(self):
        specs = make_block_specs(12, 3, overlap=0)
        self.assertEqual([s.model_index for s in specs], [0, 1, 2])
        self.assertEqual([s.noise_index for s in specs], [2, 1, 0])
        # First block has highest sigma
        self.assertGreater(specs[0].sigma_min, specs[-1].sigma_min)

    def test_layer_partition(self):
        specs = make_block_specs(12, 3, overlap=0)
        self.assertEqual(specs[0].layers, (0, 1, 2, 3))
        self.assertEqual(specs[1].layers, (4, 5, 6, 7))
        self.assertEqual(specs[2].layers, (8, 9, 10, 11))

    def test_overlap_expands_range(self):
        noise = LogNormalNoise()
        specs = make_block_specs(12, 3, noise=noise, overlap=0.2)
        for spec in specs:
            self.assertLessEqual(spec.train_sigma_min, spec.sigma_min)
            self.assertGreaterEqual(spec.train_sigma_max, spec.sigma_max)

    def test_sigma_mid(self):
        specs = make_block_specs(12, 3, overlap=0)
        for spec in specs:
            mid = spec.sigma_mid
            self.assertGreater(mid, spec.sigma_min)
            self.assertLess(mid, spec.sigma_max)


class TestSampling(unittest.TestCase):
    def test_sample_within_range(self):
        specs = make_block_specs(12, 3, overlap=0)
        rng = Random(42)
        for _ in range(100):
            block, sigma = sample_training_sigma(specs, rng=rng)
            self.assertTrue(block.contains(sigma, training=True))

    def test_edm_weight_reference(self):
        w = edm_weight(2.0, sigma_data=0.5)
        self.assertTrue(math.isclose(w, 4.25, rel_tol=1e-12))

    def test_preconditioning_reference(self):
        c = preconditioning(2.0, sigma_data=0.5)
        self.assertIsInstance(c, PrecondCoeffs)
        self.assertTrue(math.isclose(c.c_skip, 1.0 / 17.0, rel_tol=1e-12))


class TestEdmSigmas(unittest.TestCase):
    def test_descending_order(self):
        sigmas = edm_sigmas(10, descending=True)
        for i in range(len(sigmas) - 1):
            self.assertGreater(sigmas[i], sigmas[i + 1])

    def test_ascending_order(self):
        sigmas = edm_sigmas(10, descending=False)
        for i in range(len(sigmas) - 1):
            self.assertLess(sigmas[i], sigmas[i + 1])


# ============================================================
# NN module tests
# ============================================================

class TestMultiHeadAttention(unittest.TestCase):
    def test_forward_shape(self):
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = mx.random.normal((2, 10, 64))
        out = attn(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_causal_mask_shape(self):
        mask = _create_causal_mask(8)
        self.assertEqual(mask.shape, (8, 8))
        # Upper triangle should be -inf
        self.assertTrue(float(mask[0, 1]) == float("-inf"))
        self.assertTrue(float(mask[0, 0]) == 0.0)

    def test_with_causal_mask(self):
        attn = MultiHeadAttention(dim=64, num_heads=4)
        x = mx.random.normal((2, 8, 64))
        mask = _create_causal_mask(8)
        out = attn(x, mask=mask)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 8, 64))


class TestTimestepEmbedder(unittest.TestCase):
    def test_forward_shape(self):
        emb = TimestepEmbedder(hidden_size=64)
        t = mx.array([0.1, 0.5, 1.0])
        out = emb(t)
        mx.eval(out)
        self.assertEqual(out.shape, (3, 64))


class TestAdaLN(unittest.TestCase):
    def test_output_count(self):
        adaln = AdaLN(hidden_size=64)
        c = mx.random.normal((2, 64))
        chunks = adaln(c)
        self.assertEqual(len(chunks), 6)
        for chunk in chunks:
            self.assertEqual(chunk.shape, (2, 64))


class TestModulate(unittest.TestCase):
    def test_identity_at_zero(self):
        x = mx.ones((2, 5, 64))
        shift = mx.zeros((2, 64))
        scale = mx.zeros((2, 64))
        out = modulate(x, shift, scale)
        mx.eval(out)
        # (1 + 0) * 1 + 0 = 1
        self.assertTrue(mx.allclose(out, x).item())


class TestFeedForward(unittest.TestCase):
    def test_standard_mlp_shape(self):
        ff = FeedForward(dim=64, mult=4.0)
        x = mx.random.normal((2, 10, 64))
        out = ff(x)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_swiglu_shape(self):
        ff = SwiGLU(dim=64, mult=4.0)
        x = mx.random.normal((2, 10, 64))
        out = ff(x)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 10, 64))


class TestTransformerLayer(unittest.TestCase):
    def test_forward_shape(self):
        layer = DBlockTransformerLayer(dim=64, num_heads=4)
        x = mx.random.normal((2, 8, 64))
        cond = mx.random.normal((2, 64))
        mask = _create_causal_mask(8)
        out = layer(x, cond, mask=mask)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 8, 64))


class TestTransformerBlock(unittest.TestCase):
    def test_forward_shape(self):
        block = TransformerBlock(dim=64, num_heads=4, num_layers=3)
        x = mx.random.normal((2, 8, 64))
        cond = mx.random.normal((2, 64))
        out = block(x, cond)
        mx.eval(out)
        self.assertEqual(out.shape, (2, 8, 64))


# ============================================================
# Model tests
# ============================================================

class TestGPTConfig(unittest.TestCase):
    def test_presets(self):
        tiny = GPTConfig.tiny()
        self.assertEqual(tiny.depth, 6)
        self.assertEqual(tiny.num_blocks, 2)

        small = GPTConfig.small()
        self.assertEqual(small.depth, 12)


class TestDBlockGPT(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(
            vocab_size=64, dim=32, depth=4, num_heads=4,
            max_seq_len=32, num_blocks=2, dtype=mx.float32,
        )
        self.model = DBlockGPT(self.config)

    def test_forward_shape(self):
        tokens = mx.random.randint(0, 64, (2, 16))
        logits = self.model(tokens)
        mx.eval(logits)
        self.assertEqual(logits.shape, (2, 16, 64))

    def test_block_specs(self):
        self.assertEqual(len(self.model.block_specs), 2)
        self.assertEqual(self.model.block_specs[0].model_index, 0)
        self.assertEqual(self.model.block_specs[1].model_index, 1)

    def test_freeze_for_block(self):
        self.model.freeze_for_block(0)
        # Block 0 should be trainable
        tp = self.model.blocks[0].trainable_parameters()
        self.assertTrue(len(tp) > 0)

    def test_param_count(self):
        n = self.model.num_parameters()
        self.assertGreater(n, 0)

    def test_memory_estimate(self):
        mem = self.model.memory_estimate_mb()
        self.assertIn("params", mem)
        self.assertGreater(mem["params"], 0)

    def test_forward_block(self):
        tokens = mx.random.randint(0, 64, (2, 8))
        z = self.model.embed(tokens)
        cond = mx.random.normal((2, 32))
        out = self.model.forward_block(z, 0, cond)
        mx.eval(out)
        self.assertEqual(out.shape, z.shape)


# ============================================================
# Memory profiler tests
# ============================================================

class TestMemoryProfiler(unittest.TestCase):
    def test_snapshot(self):
        prof = MemoryProfiler()
        snap = prof.snapshot()
        self.assertGreater(snap.system_total_mb, 0)
        self.assertGreater(snap.process_rss_mb, 0)

    def test_summary(self):
        s = memory_summary()
        self.assertIn("RSS=", s)
        self.assertIn("Metal=", s)


# ============================================================
# Data tests
# ============================================================

class TestCharDataset(unittest.TestCase):
    def test_batch_shape(self):
        from dblocks.data.text import CharDataset
        ds = CharDataset("abcdef" * 100, seq_len=8)
        batch = ds.get_batch(4)
        self.assertEqual(batch.shape, (4, 8))

    def test_decode_roundtrip(self):
        from dblocks.data.text import CharDataset
        text = "hello world"
        ds = CharDataset(text * 100, seq_len=5)
        batch = ds.get_batch(1)
        decoded = ds.decode(batch[0].tolist())
        # decoded should be a valid substring or similar
        self.assertEqual(len(decoded), 5)
        for c in decoded:
            self.assertIn(c, text)


if __name__ == "__main__":
    unittest.main()
