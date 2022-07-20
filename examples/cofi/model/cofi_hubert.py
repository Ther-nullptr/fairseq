# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II

from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    Wav2Vec2Config
)

from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.bert.modeling_bert import (
    BertAttention, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput)

logger = logging.getLogger(__name__)

@dataclass
class CoFiHubertConfig(Wav2Vec2Config):
    hidden_size: int = field(default=768, metadata={"help": "hidden size"})
    num_attention_heads: int = field(default=12, metadata={"help": "num attention heads"})
    feature_extractor_dim: int = field(default=512, metadata={"help": "feature extractor dim"})
    num_hidden_layers: int = field(default=12, metadata={"help": "num hidden layers"})
    attention_probs_dropout_prob: float = field(default=0.1, metadata={"help": "attention probs dropout prob"})
    intermediate_size: int = field(default=3072, metadata={"help": "xxx"})
    layer_norm_eps: float = field(default=1e-12, metadata={"help": "xxx"})
    hidden_dropout_prob: float = field(default=0.1, metadata={"help": "xxx"})


@register_model("hubert_cofi", dataclass=CoFiHubertConfig)
class CoFiHubertModel(BaseFairseqModel, BertModel): # top module
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = CoFiTransformerEncoder(cfg)
        self.post_proj = CoFiPostProj(cfg)
        self.feature_extractor = ConvFeatureExtractionModel(cfg)

    def forward(
        self,
        input_raw_data=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        head_layer_z=True,
        head_z=True,
        intermediate_z=True,
        mlp_z=True,
        hidden_z=True
    ):
        device = input_raw_data.device
        with torch.no_grad():
            feature_extractor_output = self.feature_extractor(input_raw_data)
        feature_extractor_output = feature_extractor_output.transpose(1, 2)
        post_proj_output = self.post_proj(feature_extractor_output)
        post_proj_shape = post_proj_output.size()
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is None:
            attention_mask = torch.ones(post_proj_shape, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, post_proj_shape, device)
        encoder_outputs = self.encoder(
            post_proj_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class CoFiTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.layer = nn.ModuleList([CoFiBertLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions,
                intermediate_z=intermediate_z[i] if intermediate_z is not None else None,
                head_z=head_z[i] if head_z is not None else None,
                mlp_z=mlp_z[i] if mlp_z is not None else None,
                head_layer_z=head_layer_z[i] if head_layer_z is not None else None,
                hidden_z=hidden_z
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class CoFiBertLayer(BertLayer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.attention = CoFiMultiheadAttention(cfg)
        self.output = CoFiOutput(cfg)
        self.cfg = cfg

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            head_z=head_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z
        )

        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        if self.intermediate.dense is None:
            layer_output = attention_output
        else:
            self.intermediate_z = intermediate_z
            self.mlp_z = mlp_z
            self.hidden_z = hidden_z
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )
        outputs = (layer_output,) + outputs + (attention_output, )
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        if self.intermediate_z is not None:
            intermediate_output = intermediate_output.mul(self.intermediate_z)
        layer_output = self.output(
            intermediate_output, attention_output, self.mlp_z, self.hidden_z)
        return layer_output


class CoFiMultiheadAttention(BertAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.self = CoFiSelfAttention(cfg)
        self.output = CoFiBertSelfOutput(cfg)
        self.cfg = cfg

    def prune_heads(self, heads):
        len_heads = len(heads)
        if len_heads == 0:
            return

        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        if len(index) == 0:
            self.self.query = None
            self.self.key = None
            self.self.value = None
            self.output.dense = None
        else:
            self.self.query = prune_linear_layer(self.self.query, index)
            self.self.key = prune_linear_layer(self.self.key, index)
            self.self.value = prune_linear_layer(self.self.value, index)
            self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - \
            len(heads)
        self.self.all_head_size = self.self.attention_head_size * \
            self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            head_z=head_z,
        )
        attention_output = self.output(self_outputs[0], hidden_states, head_layer_z=head_layer_z, hidden_z=hidden_z)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class CoFiSelfAttention(BertSelfAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.hidden_size % cfg.num_attention_heads != 0 and not hasattr(cfg, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg.hidden_size, cfg.num_attention_heads)
            )
        self.cfg = cfg

        self.num_attention_heads = cfg.num_attention_heads
        self.attention_head_size = int(cfg.hidden_size / cfg.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(cfg.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        x_shape = x.size()
        last_dim = x_shape[-1]
        size_per_head = last_dim // self.num_attention_heads
        new_x_shape = x_shape[:-1] + (self.num_attention_heads, size_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                output_attentions=False,
                head_z=None):
        if self.value is None:
            return (None, None) if output_attentions else (None,)

        query_hidden_states = hidden_states
        mixed_query_layer = self.query(query_hidden_states)

        key_hidden_states = hidden_states
        mixed_key_layer = self.key(key_hidden_states)

        value_hidden_states = hidden_states
        mixed_value_layer = self.value(value_hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        if not hasattr(self, "ones"):
            self.ones = torch.ones(batch_size, seq_length, seq_length).float().to(
                hidden_states.device)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(mixed_value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)
        if head_z is not None:
            context_layer *= head_z

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size(
        )[:-2] + (context_layer.shape[-1] * context_layer.shape[-2],)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        return outputs


class CoFiPostProj(nn.Module):
    def __init__(self):
        self.dense = nn.Linear(512,768)

    def forward(self, feature_extractor_output, hidden_z):
        post_proj_output = self.dense(feature_extractor_output)
        if hidden_z is not None:
            post_proj_output = post_proj_output.mul(hidden_z)
        return post_proj_output


class CoFiOutput(BertOutput): # 3072 -> 768
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = CoFiLayerNorm(
            cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.cfg = cfg

    def forward(self, hidden_states, input_tensor, mlp_z, hidden_z=None, inference=False):
        hidden_states = self.dense(hidden_states)
        if mlp_z is not None:
            hidden_states *= mlp_z
        if not inference and hidden_states.sum().eq(0).item():
            return hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.dropout(hidden_states)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states

class CoFiBertSelfOutput(BertSelfOutput): # 768 -> 768 (inside the multihead attention)
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = CoFiLayerNorm(
            cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.cfg = cfg

    def forward(self, hidden_states, input_tensor, head_layer_z=None, hidden_z=None, inference=False):
        if hidden_states is None:
            return input_tensor
        hidden_states = self.dense(hidden_states)
        if head_layer_z is not None:
            hidden_states = hidden_states.mul(head_layer_z)
        if not inference and hidden_states.sum().eq(0).item():
            hidden_states = hidden_states + input_tensor
        else:
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
            hidden_states = self.LayerNorm(
                hidden_states + input_tensor, hidden_z)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states

class CoFiLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(
                input, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = input.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return output
