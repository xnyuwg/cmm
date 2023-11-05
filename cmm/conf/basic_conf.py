from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class BasicConfig:
    scheduler_type: str = 'linear'
    model_load_name: str = None
    model_save_name: str = None
    batch_size: int = 1
    data_loader_shuffle: bool = True
    lm_model_name: str = None
    device: torch.device = None
    learning_rate_slow: float = 1e-5
    learning_rate_fast: float = 1e-3
    gradient_accumulation_steps: int = None
    max_epochs: int = None
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    optimizing_no_decay: str = "bias,LayerNorm.bias,LayerNorm.weight"
    num_works: int = 0
