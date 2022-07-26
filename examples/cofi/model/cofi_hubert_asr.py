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
from fairseq.tasks import FairseqTask

from transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from .cofi_hubert import *
from .l0_module import *
from ..utils.cofi_utils import *

logger = logging.getLogger(__name__)


@dataclass
class CoFiHubertTeaStuConfig(CoFiHubertConfig):
    model_path: Optional[str] = field(
        default=None, metadata={"help": "the path to finetuned hubert model"})
    save_best_path: Optional[str] = field(
        default=None, metadata={"help": "the path to save l0 and zs"})
    label_dir: Optional[str] = field(default=None,
                                     metadata={"help": "label dir"})
    prepruning_finetune_steps: int = field(
        default=100, metadata={"help": "prepruning finetune steps"})
    hidden_size: int = field(default=768,
                             metadata={"help": "prepruning finetune steps"})
    vocab_size: int = field(default=32, metadata={"help": "vocab size"})
    do_layer_distill: bool = field(default=True,
                                   metadata={"help": "do layer distill"})
    is_decoder: bool = field(default=False, metadata={"help": "is decoder"})
    add_cross_attention: bool = field(default=False,
                                      metadata={"help": "add cross attention"})
    intermediate_size: int = field(default=3072,
                                   metadata={"help": "intermediate size"})
    num_attention_heads: int = field(default=12,
                                     metadata={"help": "num attention heads"})
    num_hidden_layers: int = field(default=12,
                                   metadata={"help": "num hidden layers"})
    lagrangian_warmup_steps: int = field(default = 200, metadata={"help": "lagrangian warmup steps"})


@register_model("cofi_hubert_tea_stu", dataclass=CoFiHubertTeaStuConfig)
class CoFiHubertTeaStu(BaseFairseqModel):
    def __init__(self, cfg: CoFiHubertTeaStuConfig,
                 student_model: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.student_model = student_model
        self.teacher_model = copy.deepcopy(student_model)
        self.l0_module = L0Module(cfg)
        self.l0_module.set_lagrangian_warmup_steps(cfg.lagrangian_warmup_steps)
        self.start_prune = False
        self.steps = 0
        self.save_best_path = cfg.save_best_path
        self.prepruning_finetune_steps = cfg.prepruning_finetune_steps
        logger.info('initialize the layer transformation')
        initialize_layer_transformation(student_model)
        logger.info(self.student_model)
        logger.info(f"Model size: {calculate_parameters(self.student_model)}")

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: CoFiHubertTeaStuConfig, task):
        student_model = CoFiHubertForASR(cfg)
        # load model with local finetuned-model
        logger.info('load finetuned model of hubert')
        ft_hubert_model = torch.load(cfg.model_path)
        ft_hubert_model = ft_hubert_model['model']
        logger.info('copy params to cofi')
        # feature_extractor
        logger.info('initial feature extractor')
        state_dict = student_model.state_dict()
        state_dict[
            'w2v_model.feature_extractor.conv_layers.0.0.weight'] = ft_hubert_model[
                'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.0.weight']
        state_dict[
            'w2v_model.feature_extractor.conv_layers.0.2.weight'] = ft_hubert_model[
                'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.2.weight']
        state_dict[
            'w2v_model.feature_extractor.conv_layers.0.2.bias'] = ft_hubert_model[
                'w2v_encoder.w2v_model.feature_extractor.conv_layers.0.2.bias']
        for i in range(1, 7):
            state_dict[
                f'w2v_model.feature_extractor.conv_layers.{i}.0.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.feature_extractor.conv_layers.{i}.0.weight']
        # post proj
        logger.info('initial post proj')
        state_dict['w2v_model.post_proj.weight'] = ft_hubert_model[
            'w2v_encoder.w2v_model.post_extract_proj.weight']
        state_dict['w2v_model.post_proj.bias'] = ft_hubert_model[
            'w2v_encoder.w2v_model.post_extract_proj.bias']
        # encoder
        logger.info('initial encoder')
        for i in range(0, 12):
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.key.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.k_proj.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.query.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.q_proj.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.value.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.v_proj.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.key.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.k_proj.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.query.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.q_proj.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.self.value.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.v_proj.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.output.dense.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.out_proj.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.output.dense.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn.out_proj.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.output.LayerNorm.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn_layer_norm.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.attention.output.LayerNorm.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.self_attn_layer_norm.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.intermediate.dense.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.fc1.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.intermediate.dense.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.fc1.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.output.dense.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.fc2.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.output.dense.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.fc2.bias']
            state_dict[
                f'w2v_model.encoder.layer.{i}.output.LayerNorm.weight'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.final_layer_norm.weight']
            state_dict[
                f'w2v_model.encoder.layer.{i}.output.LayerNorm.bias'] = ft_hubert_model[
                    f'w2v_encoder.w2v_model.encoder.layers.{i}.final_layer_norm.bias']
        # final_proj
        logger.info('initial final proj')
        state_dict['proj.weight'] = ft_hubert_model[f'w2v_encoder.proj.weight']
        state_dict['proj.bias'] = ft_hubert_model[f'w2v_encoder.proj.bias']
        return cls(cfg, student_model)

    def forward(self, raw_inputs):
        inputs = {}
        inputs['input_raw_data'] = raw_inputs['net_input']['source']
        if (1):
            self.start_prune = True
        if (self.start_prune):
            zs = self.l0_module.forward(training=True)
            self.fill_inputs_with_zs(zs, inputs)
        self.student_model.train()
        if self.l0_module is not None:
            self.l0_module.train()

        if self.teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                # ['id', 'net_input', 'target_lengths', 'ntokens', 'target']
                teacher_inputs_keys = ["input_raw_data"]
                teacher_inputs = {
                    key: inputs[key]
                    for key in teacher_inputs_keys if key in inputs
                }
                teacher_outputs = self.teacher_model(teacher_inputs)
        student_outputs = self.student_model(inputs)  #! get the two outputs

        zs = {key: inputs[key] for key in inputs if "_z" in key}
        self.steps += 1
        if (self.steps % 500 == 0 and self.start_prune):
            logger.info(f'save zs and l0_module in {self.steps}')
            zs = self.l0_module.forward(training=False)
            print(f'zs:{zs}')
            if not os.path.exists(self.save_best_path):
                os.makedirs(self.save_best_path)
            torch.save(zs, os.path.join(self.save_best_path, "zs.pt"))
            torch.save(self.l0_module,
                       os.path.join(self.save_best_path, "l0_module.pt"))
        return teacher_outputs, student_outputs, zs

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]


class CoFiHubertForASR(nn.Module):
    def __init__(self, cfg: CoFiHubertConfig):
        super().__init__()
        self.w2v_model = CoFiHubertModel(cfg)
        self.proj = CoFiLinear(cfg.hidden_size, 32)
        self.layer_transformation = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, inputs):
        loss = None
        last_hidden_state, pooled_output, hidden_states, attentions = self.w2v_model(**inputs)
        result = self.proj(pooled_output)
        return (loss, result, hidden_states, attentions)


def CoFiLinear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
