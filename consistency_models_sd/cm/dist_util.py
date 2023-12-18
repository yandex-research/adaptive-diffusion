"""
Helpers for distributed training.
"""

import os
import datetime
import torch as th
import torch.distributed as dist
from torch.distributed import barrier, all_gather, all_reduce, reduce, is_initialized


def init():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(0, 3600))
    th.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def get_rank():
    return th.distributed.get_rank() if th.distributed.is_initialized() else 0


def get_world_size():
    return th.distributed.get_world_size() if th.distributed.is_initialized() else 1


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)
