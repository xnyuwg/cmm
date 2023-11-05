from dataclasses import dataclass, fields, field
from typing import List, Dict, Tuple


@dataclass
class PDFBlockExample:
    id: str = ''
    text: str = ''
    original_text: str = ''
    size: float = 0.0
    font: str = ''
    color: int = 0
    position: List[float] = None
    page: int = 0
    xy_cut_sequence_number: int = -1
    page_height: int = 0
    page_width: int = 0

    def from_json(self, js: dict):
        for field in fields(self):
            if field.name in js:
                setattr(self, field.name, js[field.name])
        return self
