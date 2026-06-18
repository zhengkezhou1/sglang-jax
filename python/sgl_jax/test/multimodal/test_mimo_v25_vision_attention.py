"""CPU unit tests for the V-4 jit-safe MiMo-V2.5 vision attention (design §5.3.3).

The pre-V-4 form `.tolist()`-ed cu_seqlens to the host and ran a Python loop of per-image
variable-length attention -- a host readback + data-dependent trip count that cannot be jitted.
V-4 replaced it with a single batched attention masked from cu_seqlens via searchsorted (all jnp).
These tests fix the previously-pod-only validation as入库 CPU coverage:

  1. cross-segment independence: a segment-A query must NOT attend to any segment-B key (the
     block-diagonal mask), so perturbing segment-B's input leaves segment-A's output bit-identical.
  2. jit-safety: the call compiles with TRACED cu_seqlens (the old .tolist() form could not) and
     matches eager.
"""

import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=16")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh
    from jax.sharding import PartitionSpec as P

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def requires_jax(test_class):
    if not HAS_JAX:
        return unittest.skip("JAX/Flax not available")(test_class)
    return test_class


@requires_jax
class TestMiMoVisionAttentionV4(unittest.TestCase):
    HIDDEN = 32
    HEADS = 4
    HEAD_DIM = 8

    @classmethod
    def setUpClass(cls):
        tp_size = min(2, len(jax.devices()))
        cls.mesh = Mesh(
            np.asarray(jax.devices()[:tp_size]).reshape(1, tp_size),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )

    def _build(self):
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionAttention

        with jax.set_mesh(self.mesh):
            return MiMoVisionAttention(
                hidden_size=self.HIDDEN,
                num_heads=self.HEADS,
                num_kv_heads=self.HEADS,
                head_dim=self.HEAD_DIM,
                mesh=self.mesh,
                use_sink=True,
                window_size=-1,  # no windowing -> pure block-diagonal segment mask
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )

    def test_tensor_parallel_parameter_axes(self):
        attn = self._build()
        self.assertEqual(attn.q_proj.weight[...].sharding.spec, P(None, "tensor"))
        self.assertEqual(attn.k_proj.weight[...].sharding.spec, P(None, "tensor"))
        self.assertEqual(attn.v_proj.weight[...].sharding.spec, P(None, "tensor"))
        self.assertEqual(attn.proj.weight[...].sharding.spec, P("tensor", None))
        self.assertEqual(attn.sinks[...].sharding.spec, P("tensor"))

    def _identity_pos_emb(self, seq):
        # cos=1, sin=0 -> apply_rotary_pos_emb_vision is identity (q*1 + rotate_half(q)*0 = q).
        half = self.HEAD_DIM // 2
        return jnp.ones((seq, half), jnp.float32), jnp.zeros((seq, half), jnp.float32)

    def test_cross_segment_independence(self):
        attn = self._build()
        seq = 9
        cu = jnp.array([0, 4, 9], dtype=jnp.int32)  # two images: rows [0:4] and [4:9]
        cos, sin = self._identity_pos_emb(seq)
        hs = jax.random.normal(jax.random.PRNGKey(0), (seq, self.HIDDEN), jnp.float32)
        out1 = attn(hs, cu, (cos, sin), full_attn=False)
        # Perturb ONLY segment B (rows 4:9).
        hs2 = hs.at[4:].set(jax.random.normal(jax.random.PRNGKey(1), (5, self.HIDDEN), jnp.float32))
        out2 = attn(hs2, cu, (cos, sin), full_attn=False)
        # Segment A (rows 0:4) is bit-identical: cross-image keys are masked to finfo.min ->
        # softmax weight exactly 0, so segment B never leaks into segment A.
        self.assertTrue(bool(jnp.array_equal(out1[:4], out2[:4])))
        # Segment B's own output did change (sanity: the perturbation was real).
        self.assertFalse(bool(jnp.allclose(out1[4:], out2[4:])))

    def test_jit_with_traced_cu_seqlens(self):
        attn = self._build()
        seq = 9
        cu = jnp.array([0, 4, 9], dtype=jnp.int32)
        cos, sin = self._identity_pos_emb(seq)
        hs = jax.random.normal(jax.random.PRNGKey(2), (seq, self.HIDDEN), jnp.float32)
        eager = attn(hs, cu, (cos, sin), full_attn=False)

        @nnx.jit
        def run(m, h, c, cs, sn):
            return m(h, c, (cs, sn), full_attn=False)

        jitted = run(attn, hs, cu, cos, sin)  # cu is a TRACED array (no host .tolist())
        self.assertTrue(bool(jnp.allclose(eager, jitted, atol=1e-5)))

    def test_full_transformer_uses_tp_and_returns_replicated_features(self):
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionTransformer
        from sgl_jax.srt.models.mimo_v2_5.weights_mapping import (
            create_mimo_vision_weight_mappings,
        )

        config = SimpleNamespace(
            depth=1,
            hidden_size=self.HIDDEN,
            intermediate_size=64,
            num_heads=self.HEADS,
            num_key_value_heads=self.HEADS,
            qk_channels=self.HEAD_DIM,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=16,
            fullatt_block_indexes=[],
            vit_window_attn_types=[0],
            visual_token_window_size=-1,
            use_sink=False,
            hidden_act="silu",
        )
        with jax.set_mesh(self.mesh):
            model = MiMoVisionTransformer(
                config,
                mesh=self.mesh,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            output = model(jnp.ones((4, 12), dtype=jnp.float32), ((1, 2, 2),))

        self.assertEqual(output.shape, (1, 16))
        self.assertEqual(output.sharding.spec, P(None, None))
        self.assertEqual(
            model.blocks[0].mlp.gate_proj.weight[...].sharding.spec,
            P(None, "tensor"),
        )
        self.assertEqual(
            model.blocks[0].mlp.down_proj.weight[...].sharding.spec,
            P("tensor", None),
        )
        self.assertEqual(model.merger.mlp_fc1.weight[...].sharding.spec, P(None, "tensor"))
        self.assertEqual(model.merger.mlp_fc2.weight[...].sharding.spec, P("tensor", None))

        mappings = create_mimo_vision_weight_mappings(config)
        qkv = mappings["visual.blocks.0.attn.qkv.weight"]
        self.assertEqual(qkv.qkv_split_sizes, (32, 32, 32))
        self.assertEqual(qkv.sharding, (None, "tensor"))
        self.assertEqual(
            qkv.target_path,
            [
                "blocks.0.attn.q_proj.weight",
                "blocks.0.attn.k_proj.weight",
                "blocks.0.attn.v_proj.weight",
            ],
        )

        from sgl_jax.srt.utils.weight_utils import WeightLoader

        fused = jnp.arange(32 * 96, dtype=jnp.float32).reshape(32, 96)
        state = nnx.state(model)
        loader = WeightLoader(model, SimpleNamespace(), self.mesh, dtype=jnp.float32)
        loader._split_qkv_weight(
            state,
            "visual.blocks.0.attn.qkv.weight",
            fused,
            qkv,
        )
        nnx.update(model, state)
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.q_proj.weight[...], fused[:, :32]))
        )
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.k_proj.weight[...], fused[:, 32:64]))
        )
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.v_proj.weight[...], fused[:, 64:]))
        )

        qkv_bias = mappings["visual.blocks.0.attn.qkv.bias"]
        fused_bias = jnp.arange(96, dtype=jnp.float32)
        state = nnx.state(model)
        loader._split_qkv_weight(
            state,
            "visual.blocks.0.attn.qkv.bias",
            fused_bias,
            qkv_bias,
        )
        nnx.update(model, state)
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.q_proj.bias[...], fused_bias[:32]))
        )
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.k_proj.bias[...], fused_bias[32:64]))
        )
        self.assertTrue(
            bool(jnp.array_equal(model.blocks[0].attn.v_proj.bias[...], fused_bias[64:]))
        )

    def test_tp16_replicates_gqa_kv_heads(self):
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionTransformer
        from sgl_jax.srt.models.mimo_v2_5.weights_mapping import (
            create_mimo_vision_weight_mappings,
        )
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        mesh = Mesh(
            np.asarray(jax.devices()[:16]).reshape(1, 16),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        config = SimpleNamespace(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=32,
            num_key_value_heads=8,
            qk_channels=8,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=16,
            fullatt_block_indexes=[],
            vit_window_attn_types=[0],
            visual_token_window_size=-1,
            use_sink=False,
            hidden_act="silu",
        )
        with jax.set_mesh(mesh):
            model = MiMoVisionTransformer(config, mesh=mesh, dtype=jnp.float32)

        attn = model.blocks[0].attn
        self.assertEqual(attn.original_num_kv_heads, 8)
        self.assertEqual(attn.num_kv_heads, 16)
        self.assertEqual(attn.k_proj.weight.shape, (32, 16 * 8))

        mappings = create_mimo_vision_weight_mappings(config, tp_size=16)
        qkv = mappings["visual.blocks.0.attn.qkv.weight"]
        fused = jnp.arange(32 * 384, dtype=jnp.float32).reshape(32, 384)
        state = nnx.state(model)
        loader = WeightLoader(model, SimpleNamespace(), mesh, dtype=jnp.float32)
        loader._split_qkv_weight(state, "visual.blocks.0.attn.qkv.weight", fused, qkv)
        nnx.update(model, state)

        expected_k = jnp.repeat(fused[:, 256:320].reshape(32, 8, 8), 2, axis=1).reshape(32, 128)
        expected_v = jnp.repeat(fused[:, 320:384].reshape(32, 8, 8), 2, axis=1).reshape(32, 128)
        self.assertTrue(bool(jnp.array_equal(attn.k_proj.weight[...], expected_k)))
        self.assertTrue(bool(jnp.array_equal(attn.v_proj.weight[...], expected_v)))

        single_mesh = Mesh(
            np.asarray(jax.devices()[:1]).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        from sgl_jax.srt.models.mimo_v2_5.vision_encoder import MiMoVisionAttention

        with jax.set_mesh(single_mesh):
            reference = MiMoVisionAttention(
                hidden_size=32,
                num_heads=32,
                num_kv_heads=8,
                head_dim=8,
                mesh=single_mesh,
                dtype=jnp.float32,
            )

        reference.q_proj.weight[...] = fused[:, :256]
        reference.k_proj.weight[...] = fused[:, 256:320]
        reference.v_proj.weight[...] = fused[:, 320:384]
        reference.proj.weight[...] = np.asarray(attn.proj.weight[...])
        for module in (reference, attn):
            module.q_proj.bias[...] = jnp.zeros_like(module.q_proj.bias[...])
            module.k_proj.bias[...] = jnp.zeros_like(module.k_proj.bias[...])
            module.v_proj.bias[...] = jnp.zeros_like(module.v_proj.bias[...])
            module.proj.bias[...] = jnp.zeros_like(module.proj.bias[...])

        hidden_states = jnp.arange(3 * 32, dtype=jnp.float32).reshape(3, 32) / 100
        cu_seqlens = jnp.asarray([0, 3], dtype=jnp.int32)
        pos = (
            jnp.ones((3, 4), dtype=jnp.float32),
            jnp.zeros((3, 4), dtype=jnp.float32),
        )
        with jax.set_mesh(single_mesh):
            expected = reference(hidden_states, cu_seqlens, pos)
        with jax.set_mesh(mesh):
            actual = attn(hidden_states, cu_seqlens, pos)
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)))


if __name__ == "__main__":
    unittest.main()
