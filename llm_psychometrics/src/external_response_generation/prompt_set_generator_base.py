from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from datasets import load_dataset


@dataclass
class PromptSetGeneratorBase:
    source_dataset_id: str
    source_split: str = "train"
    source_revision: Optional[str] = None
    source_fingerprint: Optional[str] = None
    take_n: Optional[int] = None
    debug: bool = False
    debug_n: int = 5

    def stable_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_source_dataset(self):
        ds = load_dataset(
            self.source_dataset_id,
            split=self.source_split,
            revision=self.source_revision,
        )
        try:
            self.source_fingerprint = getattr(ds, "_fingerprint", None)
        except Exception:
            self.source_fingerprint = None
        return ds

    def compute_meta_hash(self) -> str:
        payload = {
            "source_dataset_id": self.source_dataset_id,
            "source_split": self.source_split,
            "source_revision": self.source_revision,
            "source_fingerprint": self.source_fingerprint,
            "take_n": self.take_n,
            "debug": self.debug,
            "debug_n": self.debug_n,
            "generator_class": self.__class__.__name__,
        }
        return self.stable_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def get_prompt_rows(self, source_ds) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def format_prompt(self, persona_string: str, prompt_row: Dict[str, Any]) -> str:
        raise NotImplementedError
