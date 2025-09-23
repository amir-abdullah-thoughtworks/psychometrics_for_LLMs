from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import threading
from datasets import Dataset, DatasetDict, Features, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, HfFolder, create_repo

class HFStreamingAppender:
    """
    Stream rows into a Hugging Face dataset split. Buffers rows locally and, on every
    `buffer_size` pushes, appends them to a local cached copy of the split, then pushes
    the updated split to the Hub. Designed for simple 'append-only' workflows.

    Repo layout expectation:
      - A standard Hub dataset repo with splits (e.g., 'train') already present, or
        set create_if_missing=True to bootstrap an empty split on first push.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        features: Optional[Features] = None,
        buffer_size: int = 100,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
        create_if_missing: bool = False,
        max_shard_size: str = "1GB",
        commit_prefix: str = "[stream]",
    ):
        self.repo_id = repo_id
        self.split = split
        self.features = features
        self.buffer_size = buffer_size
        self.cache_dir = cache_dir
        self.private = private
        self.max_shard_size = max_shard_size
        self.commit_prefix = commit_prefix
        self._buf: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._local_ds: Optional[Dataset] = None  # cached growing split
        self._initialized = False

        # Auth & repo bootstrap
        if hf_token:
            HfFolder.save_token(hf_token)
        self.api = HfApi(token=hf_token)

        if create_if_missing:
            # idempotent: will no-op if already exists
            create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    # --------- internal helpers ---------
    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            # Try to load existing remote split (non-streaming)
            remote = load_dataset(self.repo_id, split=self.split, cache_dir=self.cache_dir)
            self._local_ds = remote
            if self.features is None:
                self.features = remote.features
        except Exception:
            # No remote split: start empty; require features to be known
            if self.features is None:
                raise RuntimeError(
                    "Remote split not found and no `features` provided. "
                    "Pass `features` or set up the repo/split first."
                )
            empty = {k: [] for k in self.features.keys()}
            self._local_ds = Dataset.from_dict(empty).cast(self.features)
        self._initialized = True

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.features is None:
            # If features weren’t provided and no remote existed, this would have errored earlier
            return row
        # Drop extras & fill missings with None to preserve schema order
        keys = list(self.features.keys())
        out = {k: row[k] for k in keys if k in row}
        for k in keys:
            if k not in out:
                out[k] = None
        return out

    def _rows_to_dataset(self, rows: List[Dict[str, Any]]) -> Dataset:
        if not rows:
            raise ValueError("No rows to convert.")
        data = {k: [r.get(k, None) for r in rows] for k in (self.features.keys() if self.features else rows[0].keys())}
        ds = Dataset.from_dict(data)
        if self.features is not None:
            ds = ds.cast(self.features)
        return ds

    def _flush_locked(self) -> int:
        if not self._buf:
            return 0
        self._ensure_initialized()

        # Build batch dataset
        batch_ds = self._rows_to_dataset(self._buf)
        # Append to local cached split (immutably)
        self._local_ds = concatenate_datasets([self._local_ds, batch_ds])
        if self.features is not None:
            self._local_ds = self._local_ds.cast(self.features)

        # Push updated split to hub (overwrite split with new content)
        # Note: push_to_hub on a Dataset with split=... replaces that split content.
        self._local_ds.push_to_hub(
            self.repo_id,
            split=self.split,
            private=self.private,
            max_shard_size=self.max_shard_size,
            commit_message=f"{self.commit_prefix} {len(self._buf)} new rows to {self.split}",
        )

        pushed = len(self._buf)
        self._buf.clear()
        return pushed

    # --------- public API ---------
    def push(self, row: Dict[str, Any]) -> Optional[int]:
        """
        Add a single row to the in-memory buffer. When buffer reaches `buffer_size`,
        flush to the Hub. Returns number of rows flushed if a flush occurred, else None.
        """
        with self._lock:
            self._ensure_initialized()
            self._buf.append(self._normalize_row(row))
            if len(self._buf) >= self.buffer_size:
                return self._flush_locked()
        return None

    def push_many(self, rows: List[Dict[str, Any]]) -> Optional[int]:
        """
        Add multiple rows; triggers flush when threshold reached. May flush multiple times.
        Returns total flushed this call, or None if no flush occurred.
        """
        flushed_total = 0
        with self._lock:
            self._ensure_initialized()
            for r in rows:
                self._buf.append(self._normalize_row(r))
                if len(self._buf) >= self.buffer_size:
                    flushed_total += self._flush_locked()
        return flushed_total or None

    def flush(self) -> int:
        """
        Force a flush of the current buffer to the Hub regardless of size.
        Returns number of rows flushed (0 if nothing to flush).
        """
        with self._lock:
            self._ensure_initialized()
            return self._flush_locked()

    def local_count(self) -> int:
        """Current length of the local cached split (already pushed + buffered if flushed)."""
        self._ensure_initialized()
        return len(self._local_ds) if self._local_ds is not None else 0

    def buffer_count(self) -> int:
        """Rows currently waiting in memory to be flushed."""
        return len(self._buf)