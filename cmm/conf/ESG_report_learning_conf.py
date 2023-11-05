from dataclasses import dataclass
from typing import List
from cmm.conf.basic_conf import BasicConfig
from pathlib import Path


@dataclass
class ESGReportLearningConfig(BasicConfig):
    sentence_lm_embedding_dim: int = None
    sentence_token_max_length: int = None
    to_process_pipline: List[str] = None
    max_node_per_batch: int = None
    hidden_sen_emb_dim: int = None
    data_loader_read_pipeline: list = None
    common_size_chose: int = None
    data_path: Path = Path('./')
    result_path: Path = Path('./')
    cache_path: Path = Path('./')
    use_partial_data: bool = None
