from dataclasses import dataclass, fields, field
from typing import List, Dict, Tuple


@dataclass
class DocumentInfoExample:
    doc_id: str = ''
    split_full: str = ''
    split_partial: str = ''
    in_partial: bool = None

    def from_json(self, js: dict):
        for field in fields(self):
            if field.name in js:
                setattr(self, field.name, js[field.name])
        return self
