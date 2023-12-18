"""
Helpers to train with 16-bit precision.
"""

import math
import numpy as np
import torch as th

from . import logger

# For LAION experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 12-13 within the first steps of training.
INITIAL_LOG_LOSS_SCALE = 14.0


def master_params_to_state_dict(model, master_params):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict):
    master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = initial_lg_loss_scale

        self.scaler = th.cuda.amp.GradScaler(
            init_scale=2 ** INITIAL_LOG_LOSS_SCALE, 
            growth_factor=2 ** fp16_scale_growth, 
            backoff_factor=0.5, 
            growth_interval=1, 
            enabled=self.use_fp16
        )

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        self.scaler.scale(loss).backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            logger.logkv_mean("lg_loss_scale", math.log2(self.scaler.get_scale()))
            grad_norm, param_norm = self._compute_norms(grad_scale=self.scaler.get_scale())
            if check_overflow(grad_norm):
                prev_scale = self.scaler.get_scale()
                self.scaler.step(opt)
                self.scaler.update()
                assert prev_scale > self.scaler.get_scale()
                logger.log(f"Found NaN, decreased lg_loss_scale to {math.log2(self.scaler.get_scale())}")
                self.zero_grad()
                return False
        else:
            grad_norm, param_norm = self._compute_norms()

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.scaler.step(opt)
        self.scaler.update()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)
    
    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(self.model, master_params)

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
