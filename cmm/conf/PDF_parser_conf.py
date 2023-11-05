from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable


@dataclass
class PDFParserConfig:
    title_max_length: int = None

    keep_image: bool = None
    keep_image_block: bool = None
    layout_pdf_image_scale: float = None

    valid_matched_toc_ratio: float = None
    black_pp: list = None

    summarization_input_max_length: int = None
    need_for_summarization_min_length: int = None

    use_common_size: bool = None
    hier_keep_all: bool = None
    processing_hier: bool = False
