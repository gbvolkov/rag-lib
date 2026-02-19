from typing import Any, Dict, Iterable, List, Optional
import re
import uuid

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.chunkers.base import TextSplitter
from rag_lib.chunkers.table_rows import (
    build_summary_content,
    chunk_table_rows,
    parse_markdown_table,
    render_markdown_table,
)
from rag_lib.config import Settings
from rag_lib.summarizers.table import TableSummarizer

class MarkdownTableSplitter(TextSplitter):
    """
    Splits text into Table Segments and Text Segments based on Markdown GFM table syntax.
    Inherits from TextSplitter to support standardized pipeline.
    """
    def __init__(
        self,
        *,
        split_table_rows: bool = False,
        max_rows_per_table_chunk: Optional[int] = None,
        max_table_chunk_size: Optional[int] = None,
        summarizer: Optional[TableSummarizer] = None,
        summarize_table: bool = True,
        summarize_chunks: bool = False,
        inject_summaries_into_content: bool = False,
    ):
        settings = Settings()
        resolved_rows_per_chunk = (
            settings.ingestion.chunk_size
            if max_rows_per_table_chunk is None
            else max_rows_per_table_chunk
        )

        super().__init__(chunk_size=resolved_rows_per_chunk, chunk_overlap=0)
        self.split_table_rows = split_table_rows
        self.max_rows_per_table_chunk = resolved_rows_per_chunk
        self.max_table_chunk_size = max_table_chunk_size
        self.summarizer = summarizer
        self.summarize_table = summarize_table
        self.summarize_chunks = summarize_chunks
        self.inject_summaries_into_content = inject_summaries_into_content

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
        process_table_segments = (
            self.split_table_rows
            or self.inject_summaries_into_content
            or (self.summarizer is not None and (self.summarize_table or self.summarize_chunks))
        )

        for seg in segments:
            if seg.type == SegmentType.TEXT:
                # Try to find tables in this text segment
                sub_segments = self._parse(seg.content, seg.metadata)
                # Link parent
                for sub in sub_segments:
                    sub.parent_id = seg.segment_id
                    sub.path = seg.path
                    sub.level = seg.level
                    final_segments.append(sub)
            elif seg.type == SegmentType.TABLE and process_table_segments:
                sub_segments = self._emit_table_segments(seg.content, seg.metadata)
                for sub in sub_segments:
                    sub.parent_id = seg.segment_id
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
            
        segments: List[Segment] = []
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
            segments.extend(self._emit_table_segments(table_content, metadata))
            
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
        for index, segment in enumerate(segments):
            segment.metadata["chunk_index"] = index
            segment.metadata["chunk_total"] = total
             
        return segments

    def _emit_table_segments(
        self,
        table_content: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[Segment]:
        meta_base = metadata.copy() if metadata else {}
        parsed_table = parse_markdown_table(table_content)

        if self.split_table_rows and parsed_table.header:
            if not parsed_table.rows:
                # Header-only tables are ignored in row-split mode by contract.
                return []

            table_summary = self._summarize_markdown(
                render_markdown_table(parsed_table.header, parsed_table.rows),
                summary_level="table",
            )
            row_chunks = chunk_table_rows(
                header=parsed_table.header,
                rows=parsed_table.rows,
                render_chunk=render_markdown_table,
                max_rows_per_chunk=self.max_rows_per_table_chunk,
                max_chunk_size=self.max_table_chunk_size,
                length_function=self._length_function,
            )

            total_chunks = len(row_chunks)
            split_segments: List[Segment] = []
            for table_chunk_index, row_chunk in enumerate(row_chunks):
                original_content = render_markdown_table(parsed_table.header, row_chunk.rows)
                chunk_summary = self._summarize_markdown(
                    original_content,
                    summary_level="chunk",
                )
                final_content = original_content
                if self.inject_summaries_into_content:
                    final_content = build_summary_content(
                        original_content,
                        table_summary=table_summary,
                        chunk_summary=chunk_summary,
                    )

                meta_table = meta_base.copy()
                meta_table.update(
                    {
                        "is_table": True,
                        "table_format": "markdown",
                        "split_strategy": self.__class__.__name__,
                        "table_chunk_index": table_chunk_index,
                        "table_chunk_total": total_chunks,
                        "data_row_start": row_chunk.data_row_start,
                        "data_row_end": row_chunk.data_row_end,
                        "data_row_count": len(row_chunk.rows),
                    }
                )
                if table_summary:
                    meta_table["table_summary"] = table_summary
                if chunk_summary:
                    meta_table["chunk_summary"] = chunk_summary
                if self.inject_summaries_into_content:
                    meta_table["original_content"] = original_content

                split_segments.append(
                    Segment(
                        content=final_content,
                        type=SegmentType.TABLE,
                        original_format="markdown",
                        metadata=meta_table,
                        segment_id=str(uuid.uuid4()),
                    )
                )
            return split_segments

        # Default mode: keep full table block as one segment and optionally summarize/inject.
        table_summary = self._summarize_markdown(table_content, summary_level="table")
        chunk_summary = self._summarize_markdown(table_content, summary_level="chunk")

        final_content = table_content
        if self.inject_summaries_into_content:
            final_content = build_summary_content(
                table_content,
                table_summary=table_summary,
                chunk_summary=chunk_summary,
            )

        meta_table = meta_base.copy()
        meta_table.update(
            {
                "is_table": True,
                "table_format": "markdown",
                "split_strategy": self.__class__.__name__,
                "table_chunk_index": 0,
                "table_chunk_total": 1,
            }
        )

        if parsed_table.header:
            meta_table["data_row_start"] = 0
            meta_table["data_row_end"] = len(parsed_table.rows)
            meta_table["data_row_count"] = len(parsed_table.rows)

        if table_summary:
            meta_table["table_summary"] = table_summary
        if chunk_summary:
            meta_table["chunk_summary"] = chunk_summary
        if self.inject_summaries_into_content:
            meta_table["original_content"] = table_content

        return [
            Segment(
                content=final_content,
                type=SegmentType.TABLE,
                original_format="markdown",
                metadata=meta_table,
                segment_id=str(uuid.uuid4()),
            )
        ]

    def _summarize_markdown(self, markdown_table: str, summary_level: str) -> Optional[str]:
        if not self.summarizer:
            return None
        if summary_level == "table" and not self.summarize_table:
            return None
        if summary_level == "chunk" and not self.summarize_chunks:
            return None
        if not markdown_table.strip():
            return None

        try:
            return self.summarizer.summarize(markdown_table)
        except Exception:
            return None
