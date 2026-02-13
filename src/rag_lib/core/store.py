import json
import os
import pickle
from typing import Iterator, List, Optional, Sequence, Tuple, Any

from langchain_core.stores import BaseStore
from rag_lib.core.domain import Segment

class JsonFileStore(BaseStore[str, Segment]):
    """
    A simple JSON-based store for Segments. 
    Not performant for millions of records, but great for debugging and small datasets.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self._data = raw_data
            except (json.JSONDecodeError, ValueError):
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        # atomic write potential here, but for now simple write
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def mget(self, keys: Sequence[str]) -> List[Optional[Segment]]:
        results = []
        for k in keys:
            raw = self._data.get(k)
            if raw:
                # Deserialize dict -> Segment
                results.append(Segment(**raw))
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Segment]]) -> None:
        for k, v in key_value_pairs:
            # Serialize Segment -> dict
            self._data[k] = v.model_dump() # Pydantic v2
        self._save()

    def mdelete(self, keys: Sequence[str]) -> None:
        changed = False
        for k in keys:
            if k in self._data:
                del self._data[k]
                changed = True
        if changed:
            self._save()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        for k in self._data.keys():
            if prefix is None or k.startswith(prefix):
                yield k

class LocalPickleStore(BaseStore[str, Segment]):
    """
    A simple Pickle-based store for Segments.
    Faster for binary data, but not human readable.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: dict[str, Segment] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "rb") as f:
                    self._data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        dir_name = os.path.dirname(self.file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(self.file_path, "wb") as f:
            pickle.dump(self._data, f)

    def mget(self, keys: Sequence[str]) -> List[Optional[Segment]]:
        return [self._data.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Segment]]) -> None:
        for k, v in key_value_pairs:
            self._data[k] = v
        self._save()

    def mdelete(self, keys: Sequence[str]) -> None:
        changed = False
        for k in keys:
            if k in self._data:
                del self._data[k]
                changed = True
        if changed:
            self._save()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        for k in self._data.keys():
            if prefix is None or k.startswith(prefix):
                yield k
