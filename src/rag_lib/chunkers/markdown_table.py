from typing import List, Callable, Iterable, Optional, Dict, Any
import re
import uuid
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.chunkers.base import TextSplitter

class MarkdownTableSplitter(TextSplitter):
    """
    Splits text into Table Segments and Text Segments based on Markdown GFM table syntax.
    Inherits from TextSplitter to support standardized pipeline.
    """
    def __init__(self):
        super().__init__()
        # Improved Regex for GFM tables
        self.table_pattern = re.compile(
            r'('
            r'(?:^.*?\|.*(?:\n|$))'    # Header Row
            r'(?:\s*\|[-:| ]+\|\s*(?:\n|$))' # Separator Row
            r'(?:.*?\|.*(?:\n|$))*'    # Body Rows
            r')',
            re.MULTILINE
        )

    def split_text(self, text: str) -> List[str]:
        """
        Required by base class. Returns raw content strings.
        Note: This loses type information (Table vs Text). 
        To get typed segments, use create_segments or split_segments.
        """
        segments = self._parse(text)
        return [s.content for s in segments]

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        """
        Override to return typed segments (Table/Text).
        """
        return self._parse(text, metadata)

    def split_segments(self, segments: Iterable[Segment]) -> List[Segment]:
        """
        Refines segments. If a segment is TEXT, tries to find recursive tables.
        If TABLE, keeps as is.
        """
        final_segments = []
        for seg in segments:
            if seg.type == SegmentType.TEXT:
                # Try to find tables in this text segment
                sub_segments = self._parse(seg.content, seg.metadata)
                # Link parent
                for sub in sub_segments:
                     sub.parent_id = seg.segment_id
                     # Inherit other hierarchy props if needed?
                     sub.path = seg.path
                     sub.level = seg.level
                     final_segments.append(sub)
            else:
                # Keep non-text segments (e.g. already TABLE or IMAGE) as is
                final_segments.append(seg)
        return final_segments

    def _parse(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        if metadata is None:
            metadata = {}
            
        segments = []
        last_pos = 0
        
        matches = list(self.table_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            start, end = match.span()
            
            # Text before table
            if start > last_pos:
                text_content = text[last_pos:start].strip()
                if text_content:
                    meta = metadata.copy()
                    meta.update({
                        "chunk_index": len(segments),
                        "split_strategy": self.__class__.__name__
                    })
                    segments.append(Segment(
                        content=text_content,
                        type=SegmentType.TEXT,
                        original_format="text",
                        metadata=meta,
                        segment_id=str(uuid.uuid4())
                    ))
            
            # Table Segment
            table_content = match.group(0).strip()
            meta_table = metadata.copy()
            meta_table.update({
                "chunk_index": len(segments),
                "is_table": True,
                "split_strategy": self.__class__.__name__
            })
            segments.append(Segment(
                content=table_content,
                type=SegmentType.TABLE,
                original_format="markdown",
                metadata=meta_table,
                segment_id=str(uuid.uuid4())
            ))
            
            last_pos = end
            
        # Remaining text
        if last_pos < len(text):
            text_content = text[last_pos:].strip()
            if text_content:
                meta = metadata.copy()
                meta.update({
                    "chunk_index": len(segments),
                    "split_strategy": self.__class__.__name__
                })
                segments.append(Segment(
                    content=text_content,
                    type=SegmentType.TEXT,
                    original_format="text",
                    metadata=meta,
                    segment_id=str(uuid.uuid4())
                ))
                
        # If nothing found, but text exists, return as one TEXT segment
        if not segments and text.strip():
             meta = metadata.copy()
             meta.update({"split_strategy": self.__class__.__name__})
             return [Segment(content=text, type=SegmentType.TEXT, metadata=meta, segment_id=str(uuid.uuid4()))]
        
        # Update chunk_total
        total = len(segments)
        for s in segments:
            s.metadata["chunk_total"] = total
             
        return segments
