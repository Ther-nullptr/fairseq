# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class CofiAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default='(0.9, 0.999)', metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")
    learning_rate: float = field(
        default=2e-5, metadata={"help": "learning rate"}
    )
    reg_learning_rate: float = field(
        default=0.1, metadata={"help": "reg learning rate"}
    )


@register_optimizer("cofi_adam", dataclass=CofiAdamConfig)
class CofiAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: CofiAdamConfig, params):
        super().__init__(cfg)
        self.cfg = cfg

        main_model_params = [
            {
                "params": params["main_model_params_1"],
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.learning_rate
            },
            {
                "params": params["main_model_params_2"],
                "weight_decay": 0.0,
                "lr": self.cfg.learning_rate
            },
        ]

        l0_params = [{
            "params": params['l0_params'],
            "weight_decay":
            0.0,
            "lr":
            self.cfg.reg_learning_rate
        }]

        lagrangian_params = [{
            "params": params['lagrangian_params'],
            "weight_decay":
            0.0,
            "lr":
            -self.cfg.reg_learning_rate
        }]

        self._optimizer = torch.optim.Adam(
            main_model_params + l0_params + lagrangian_params,
            betas = eval(self.cfg.adam_betas),
            eps = self.cfg.adam_eps
        )

        print(self._optimizer)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)
