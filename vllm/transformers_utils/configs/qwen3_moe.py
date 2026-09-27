# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MoE configuration compatibility for older Transformers versions."""

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation

try:
    from transformers import Qwen3MoeConfig
except ImportError:

    class Qwen3MoeConfig(PretrainedConfig):
        """Backport of the configuration fields in Transformers 4.51.3.

        Keeping this config local allows existing LLMStation installations to
        load Qwen3 MoE checkpoints without upgrading their Transformers stack.
        Newer Transformers installations use their native configuration class.
        """

        model_type = "qwen3_moe"
        keys_to_ignore_at_inference = ["past_key_values"]

        def __init__(
            self,
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=None,
            hidden_act="silu",
            max_position_embeddings=32768,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            use_sliding_window=False,
            sliding_window=4096,
            max_window_layers=28,
            attention_dropout=0.0,
            decoder_sparse_step=1,
            moe_intermediate_size=768,
            num_experts_per_tok=8,
            num_experts=128,
            norm_topk_prob=False,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            mlp_only_layers=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            # Qwen3's attention projection width need not equal hidden_size.
            self.head_dim = (hidden_size // num_attention_heads
                             if head_dim is None else head_dim)
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.rope_theta = rope_theta
            self.rope_scaling = (None if rope_scaling is None else
                                 dict(rope_scaling))
            self.attention_bias = attention_bias
            self.use_sliding_window = use_sliding_window
            self.sliding_window = (sliding_window
                                   if use_sliding_window else None)
            self.max_window_layers = max_window_layers
            self.attention_dropout = attention_dropout
            self.decoder_sparse_step = decoder_sparse_step
            self.moe_intermediate_size = moe_intermediate_size
            self.num_experts_per_tok = num_experts_per_tok
            self.num_experts = num_experts
            self.norm_topk_prob = norm_topk_prob
            self.output_router_logits = output_router_logits
            self.router_aux_loss_coef = router_aux_loss_coef
            self.mlp_only_layers = ([] if mlp_only_layers is None else
                                    list(mlp_only_layers))
            if self.rope_scaling is not None and "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling["type"]

            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
            rope_config_validation(self)


__all__ = ["Qwen3MoeConfig"]
