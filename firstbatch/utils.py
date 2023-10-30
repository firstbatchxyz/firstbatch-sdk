from __future__ import annotations
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config(DataClassJsonMixin):
    """
    Configuration for the vector store.
    """
    batch_size: Optional[int] = None
    quantizer_train_size: Optional[int] = None
    quantizer_type: Optional[str] = None
    enable_history: Optional[bool] = None
    verbose: Optional[bool] = None
