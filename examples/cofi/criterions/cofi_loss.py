# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import sys
import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import logging
import re

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
import wandb

logger = logging.getLogger(__name__)

@dataclass
class CoFiCriterionConfig(FairseqDataclass):
    prepruning_finetune_steps: int = field(
        default=100,
        metadata={
            "help": "begin pruning"
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "gradient accumulation steps"
        },
    )
    distill_temp: float = field(
        default=2.0,
        metadata={
            "help": "distill temp"
        },
    )
    target_sparsity: float = field(
        default=0.95,
        metadata={
            "help": "target sparsity"
        },
    )
    distill_ce_loss_alpha: float = field(
        default=5,
        metadata={
            "help": "distill ce loss alpha"
        },
    )
    distill_loss_alpha: float = field(
        default=5,
        metadata={
            "help": "distill loss alpha"
        },
    )
    do_layer_distill: bool = field(
        default=True,
        metadata={
            "help": "do layer distill"
        },
    )
    layer_distill_version: int = field(
        default=4,
        metadata={
            "help": "layer distill version"
        },
    )

@register_criterion("cofi_loss", dataclass=CoFiCriterionConfig)
class CoFiCriterion(FairseqCriterion):
    def __init__(self, cfg: CoFiCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.steps = 0
        self.start_prune = False
        self.prepruning_finetune_steps = cfg.prepruning_finetune_steps
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.distill_temp = cfg.distill_temp
        self.distill_ce_loss_alpha = cfg.distill_ce_loss_alpha
        self.distill_loss_alpha = cfg.distill_loss_alpha
        self.do_layer_distill = cfg.do_layer_distill
        self.layer_distill_version = cfg.layer_distill_version
        logger.info('initial CoFiCriterion')

    def forward(self, model, sample):
        teacher_outputs, student_outputs, zs = model(sample)
        distill_loss = None
        distill_ce_loss = None
        distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(model, teacher_outputs, student_outputs, zs)

        lagrangian_loss = None
        if model.start_prune:
            lagrangian_loss, _, _ = \
                model.l0_module.lagrangian_regularization(self.steps - self.prepruning_finetune_steps)
            loss += lagrangian_loss

        if self.gradient_accumulation_steps > 1: # TODO
            loss = loss / self.gradient_accumulation_steps
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else None
        )

        sample_size = sample["target"].size(0)
        print(f'loss:{loss}, lagrangian_loss:{lagrangian_loss}, distill_layer_loss:{distill_loss}, distill_ce_loss:{distill_ce_loss}')
        
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "distill_loss": float(distill_loss),
            "distill_ce_loss": float(distill_ce_loss),
            "lagrangian_loss": None if lagrangian_loss == None else float(lagrangian_loss)
        }
        self.steps += 1
        return loss, sample_size, logging_output
            
    def calculate_distillation_loss(self, model, teacher_outputs, student_outputs, zs):
        layer_loss = self.calculate_layer_distillation_loss(model, teacher_outputs, student_outputs, zs)
        distill_loss = layer_loss
        prob_size = teacher_outputs[1].shape
        teacher_logits = teacher_outputs[1]
        student_logits = student_outputs[1]
        teacher_logits = teacher_logits.view(prob_size[0] * prob_size[1], prob_size[2])
        student_logits = student_logits.view(prob_size[0] * prob_size[1], prob_size[2])
        ce_distill_loss = F.kl_div(
            input=F.log_softmax(
                student_logits / self.distill_temp, dim=-1), #! logits: [32,3]
            target=F.softmax(
                teacher_logits / self.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean") * (self.distill_temp ** 2)
        loss = self.distill_ce_loss_alpha * ce_distill_loss
        if distill_loss is not None:
            loss += self.distill_loss_alpha * distill_loss
        wandb.log({'distill_loss':distill_loss,'ce_distill_loss':ce_distill_loss,'loss':loss})
        return distill_loss, ce_distill_loss, loss

    def calculate_layer_distillation_loss(self, model, teacher_outputs, student_outputs, zs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        if self.do_layer_distill: #! only do layer distill
            mlp_z = None
            head_layer_z = None
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            teacher_layer_output = teacher_outputs[2][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            student_layer_output = student_outputs[2][1:] 

            # distilliting existing layers
            if self.layer_distill_version == 2:
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    s_layer_o = model.student_model.layer_transformation(s_layer_o)
                    l = mse_loss(t_layer_o, s_layer_o)
                    if mlp_z[layer_num] > 0:
                        layer_loss += l

            # distilling layers with a minimal distance
            elif self.layer_distill_version > 2:
                l = []
                specified_teacher_layers = [2, 5, 8, 11]
                transformed_s_layer_o = [model.student_model.layer_transformation(
                    s_layer_o) for s_layer_o in student_layer_output]
                specified_teacher_layer_reps = [
                    teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

                device = transformed_s_layer_o[0].device
                for t_layer_o in specified_teacher_layer_reps:
                    for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
                        l.append(mse_loss(t_layer_o, s_layer_o))
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0
                    existing_layers = existing_layers.to(layerwiseloss.device)

                layer_loss = 0
                #! no ordering restriction specified
                if self.layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                #! added the ordering restriction -> to choose the min loss in 4 student layers
                elif self.layer_distill_version == 4:
                    last_aligned_layer = 12
                    alignment = []
                    for search_index in range(3, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{self.layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(4).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                if self.steps % 100 == 0:
                    logger.info(f"v{self.layer_distill_version} Global step: {self.steps}, Alignment: " + str(alignment))
            return layer_loss
        else:
            return None

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])