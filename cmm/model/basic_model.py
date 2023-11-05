import logging
from torch import nn
from transformers import AutoModel
import torch.nn.utils
from cmm.conf.global_config_manager import GlobalConfigManager
import math
import torch.nn.functional as F
from cmm.conf.basic_conf import BasicConfig
from transformers import AutoTokenizer, PreTrainedTokenizer


class BasicModel(nn.Module):
    @staticmethod
    def new_auto_model(model_name: str) -> AutoModel:
        model = AutoModel.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return model

    def __init__(self,
                 config: BasicConfig = None,
                 gradient_accumulation_steps: int = None):
        super().__init__()
        self.config = config
        self.slow_para = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.tokenizer = {}

    def get_model_device(self):
        return self.dummy_param.device

    def get_tokenizer(self, model_name: str = None) -> PreTrainedTokenizer:
        if model_name is None:
            model_name = self.config.lm_model_name
        if model_name not in self.tokenizer:
            logging.info("init {} auto tokenizer".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            self.tokenizer[model_name] = tokenizer
        return self.tokenizer[model_name]
