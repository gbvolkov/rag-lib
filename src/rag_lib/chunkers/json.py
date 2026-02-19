from typing import List, Optional, Any, Dict, Union
import json
import uuid
from rag_lib.chunkers.base import TextSplitter
from rag_lib.core.domain import Segment, SegmentType

class JsonSplitter(TextSplitter):
    """
    Splits a JSON string into segments.
    Primarily designed to split a top-level list into individual item segments.
    """
    def __init__(
        self,
        min_chunk_size: int = 0, # Unused for JSON logic usually, but kept for interface
        jq_schema: str = ".",
        ensure_ascii: bool = False,
    ):
        super().__init__()
        self.jq_schema = jq_schema
        self.ensure_ascii = ensure_ascii

    def split_text(self, text: str) -> List[str]:
        """
        Parses JSON and returns list of JSON strings for each item.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [] # Fail gracefully or raise? Return [] implies no splits.

        # Apply schema (Simplified)
        target = data
        if self.jq_schema != ".":
            # Very basic support for key access e.g. .items
            keys = self.jq_schema.strip(".").split(".")
            for k in keys:
                if k and isinstance(target, dict) and k in target:
                    target = target[k]
                elif k:
                    # Key not found
                    return []
        
        if isinstance(target, list):
            chunks = []
            for item in target:
                if isinstance(item, (dict, list)):
                    chunks.append(
                        json.dumps(item, indent=2, ensure_ascii=self.ensure_ascii)
                    )
                else:
                    chunks.append(str(item))
            return chunks
            
        elif isinstance(target, dict):
            # Treat whole dict as one chunk? Or split keys?
            # Defaulting to one chunk if it's a single dict object
            return [
                json.dumps(target, indent=2, ensure_ascii=self.ensure_ascii)
            ]
            
        return [str(target)]

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        """
        Override to allow customized JSON metadata or type?
        For now, standard behavior is fine, but we ensure content is valid JSON string.
        """
        if metadata is None:
            metadata = {}
            
        chunks = self.split_text(text)
        segments = []
        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.count('\n') + 1
            meta = metadata.copy()
            meta["json_index"] = i
            
            segments.append(Segment(
                content=chunk,
                segment_id=str(uuid.uuid4()),
                type=SegmentType.TEXT, # Or CODE? Or DATA? Keeping TEXT for now.
                start_index=0, # Hard to track character index in original file after json.loads
                end_index=len(chunk),
                metadata=meta
            ))
            
        return segments
