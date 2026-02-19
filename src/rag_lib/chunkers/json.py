from typing import Any, Dict, List, Optional
import json
import uuid
from rag_lib.chunkers.base import TextSplitter
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.data_loaders import SchemaDialect, iter_json_leaf_paths, select_schema_target

class JsonSplitter(TextSplitter):
    """
    Splits a JSON string into segments.
    Primarily designed to split a top-level list into individual item segments.
    """
    def __init__(
        self,
        min_chunk_size: int = 0, # Unused for JSON logic usually, but kept for interface
        schema: str = ".",
        schema_dialect: SchemaDialect = SchemaDialect.DOT_PATH,
        ensure_ascii: bool = False,
        metadata_value_max_len: Optional[int] = 256,
    ):
        super().__init__()
        if isinstance(schema_dialect, str):
            schema_dialect = SchemaDialect(schema_dialect)
        if metadata_value_max_len is not None and metadata_value_max_len < 0:
            raise ValueError("metadata_value_max_len must be >= 0 or None.")

        normalized_schema = str(schema).strip() if schema is not None else "."
        self.schema = normalized_schema if normalized_schema else "."
        self.schema_dialect = schema_dialect
        self.ensure_ascii = ensure_ascii
        self.metadata_value_max_len = metadata_value_max_len

    def split_text(self, text: str) -> List[str]:
        """
        Parses JSON and returns list of JSON strings for each item.
        """
        items = self._extract_items(text)
        return [self._serialize_item(item) for item in items]

    def _extract_items(self, text: str) -> List[Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        found, target = select_schema_target(data, self.schema, self.schema_dialect)
        if not found:
            return []

        if isinstance(target, list):
            return list(target)
        return [target]

    def _serialize_item(self, item: Any) -> str:
        if isinstance(item, (dict, list)):
            return json.dumps(item, indent=2, ensure_ascii=self.ensure_ascii)
        return str(item)

    def _truncate_string(self, value: str) -> str:
        max_len = self.metadata_value_max_len
        if max_len is None:
            return value
        if len(value) <= max_len:
            return value
        if max_len <= 3:
            return value[:max_len]
        return value[: max_len - 3] + "..."

    def _normalize_metadata_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._truncate_string(value)
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return self._truncate_string(str(value))

    def _extract_json_metadata(self, item: Any) -> Dict[str, Any]:
        promoted: Dict[str, Any] = {}
        for path, value in iter_json_leaf_paths(item):
            key_parts = ("json",) + path if path else ("json", "value")
            key = "__".join(key_parts)
            promoted[key] = self._normalize_metadata_value(value)
        return promoted

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        """
        Creates one segment per selected JSON item and promotes JSON fields
        into filterable metadata keys.
        """
        if metadata is None:
            metadata = {}

        items = self._extract_items(text)
        segments: List[Segment] = []
        for i, item in enumerate(items):
            chunk = self._serialize_item(item)
            meta = metadata.copy()
            meta["json_index"] = i
            meta.update(self._extract_json_metadata(item))

            segments.append(Segment(
                content=chunk,
                segment_id=str(uuid.uuid4()),
                type=SegmentType.TEXT, # Or CODE? Or DATA? Keeping TEXT for now.
                metadata=meta
            ))

        return segments
