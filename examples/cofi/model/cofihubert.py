# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)

from transformers.models.bert.modeling_bert import (
    BertAttention, BertEmbeddings, BertEncoder, BertForQuestionAnswering,
    BertForSequenceClassification, BertLayer, BertModel, BertOutput,
    BertSelfAttention, BertSelfOutput, QuestionAnsweringModelOutput)

logger = logging.getLogger(__name__)

@dataclass
class HubertCofiConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

@register_model("hubert_cofi", dataclass=HubertCofiConfig)
class HubertCofiModel(BaseFairseqModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = CofiTransformerEncoder(config)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        head_layer_z=None,
        head_z=None,
        intermediate_z=None,
        mlp_z=None,
        hidden_z=None
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        feature_extractor_output = self.feature_extractor(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, hidden_z=hidden_z
        )
        encoder_outputs = self.encoder(
            feature_extractor_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            intermediate_z=intermediate_z,
            head_z=head_z,
            mlp_z=mlp_z,
            head_layer_z=head_layer_z,
            hidden_z=hidden_z
        )

        sequence_output = encoder_outputs[0]
        
        return {"x":sequence_output,"featrues":feature_extractor_output}
    
class CofiPostExtractProj(nn.Module):
    def __init__(self, embed, encoder_embed_dim):
        pass
    
    def forward(x):
        pass
    
class CofiTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([CoFiBertLayer(config)
                                   for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
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
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
    
class CofiTransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CoFiBertAttention(config)
        self.output = CoFiBertOutput(config)
        self.layer_norm = CoFiLayerNorm(config)
        self.dropout = nn.Dropout()
        self.config = config

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

    
class CoFiBertLayer(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = CoFiMultiheadAttention(config)
        self.output = CoFiOutput(config)
        self.config = config

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
    def __init__(self, config):
        super().__init__(config)
        self.self = CoFiSelfAttention(config)
        self.output = CoFiOutput(config)
        self.config = config

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
            self.output.dense = prune_linear_layer(
                self.output.dense, index, dim=1)

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

        attention_output = self.output(
            self_outputs[0], hidden_states, head_layer_z=head_layer_z, hidden_z=hidden_z)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class CoFiSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

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


class CofiIntermediate(nn.Module): # 768 -> 3072
    def __init__(self):
        pass
    def forward(x):
        pass

class CoFiOutput(BertOutput): # 3072 -> 768
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = CoFiLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

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
            hidden_states = self.LayerNorm(
                hidden_states + input_tensor, hidden_z)
            if hidden_z is not None:
                hidden_states = hidden_states.mul(hidden_z)
        return hidden_states

class CoFiBertSelfOutput(BertSelfOutput): # 768 -> 768 (inside the multihead attention)
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = CoFiLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

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
    
@register_model("hubert_ctc", dataclass=HubertCtcConfig)
class CoFiHubertCtc(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x
    
class CoFiHubertEncoder():
    def __init__(self, cfg: HubertAsrConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if task.target_dictionary is not None:
            self.proj = CoFiFinalLinear(d, len(task.target_dictionary))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = CoFiFinalLinear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

def CoFiFinalLinear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m