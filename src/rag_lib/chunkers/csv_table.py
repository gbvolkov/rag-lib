from typing import Any, Callable, Dict, Iterable, List, Optional
import uuid

from rag_lib.chunkers.base import TextSplitter
from rag_lib.chunkers.table_rows import (
    build_summary_content,
    chunk_table_rows,
    parse_csv_table,
    render_csv_table,
    render_markdown_table,
)
from rag_lib.config import Settings
from rag_lib.core.domain import Document, Segment, SegmentType
from rag_lib.summarizers.table import TableSummarizer


class CSVTableSplitter(TextSplitter):
    """
    Splits CSV table text into row-based table chunks.
    Each output chunk includes the header row plus one or more data rows.
    Header handling is configurable via `use_first_row_as_header`.
    """

    def __init__(
        self,
        max_rows_per_chunk: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        delimiter: Optional[str] = None,
        use_first_row_as_header: bool = True,
        summarizer: Optional[TableSummarizer] = None,
        summarize_table: bool = True,
        summarize_chunks: bool = False,
        inject_summaries_into_content: bool = False,
        length_function: Callable[[str], int] = len,
    ):
        settings = Settings()
        resolved_rows_per_chunk = (
            settings.ingestion.chunk_size
            if max_rows_per_chunk is None
            else max_rows_per_chunk
        )
        super().__init__(chunk_size=resolved_rows_per_chunk, chunk_overlap=0, length_function=length_function)

        self.max_rows_per_chunk = resolved_rows_per_chunk
        self.max_chunk_size = max_chunk_size
        self.delimiter = delimiter
        self.use_first_row_as_header = use_first_row_as_header
        self.summarizer = summarizer
        self.summarize_table = summarize_table
        self.summarize_chunks = summarize_chunks
        self.inject_summaries_into_content = inject_summaries_into_content

    def split_text(self, text: str) -> List[str]:
        segments = self.create_segments(text)
        return [segment.content for segment in segments]

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        base_metadata = metadata.copy() if metadata else {}
        parsed_table, delimiter = parse_csv_table(
            text,
            delimiter=self.delimiter,
            use_first_row_as_header=self.use_first_row_as_header,
        )
        if not parsed_table.header or not parsed_table.rows:
            return []

        table_summary = self._summarize_table(parsed_table.header, parsed_table.rows)
        row_chunks = chunk_table_rows(
            header=parsed_table.header,
            rows=parsed_table.rows,
            render_chunk=lambda h, r: render_csv_table(h, r, delimiter=delimiter),
            max_rows_per_chunk=self.max_rows_per_chunk,
            max_chunk_size=self.max_chunk_size,
            length_function=self._length_function,
        )
        return self._build_segments(
            header=parsed_table.header,
            row_chunks=row_chunks,
            delimiter=delimiter,
            metadata=base_metadata,
            table_summary=table_summary,
        )

    def split_documents(self, documents: Iterable[Document]) -> List[Segment]:
        all_segments: List[Segment] = []
        for doc in documents:
            all_segments.extend(self.create_segments(doc.page_content, metadata=doc.metadata))
        return all_segments

    def split_segments(self, segments: Iterable[Segment]) -> List[Segment]:
        refined: List[Segment] = []
        for segment in segments:
            if segment.type != SegmentType.TABLE:
                refined.append(segment)
                continue

            sub_segments = self.create_segments(segment.content, metadata=segment.metadata)
            for sub_segment in sub_segments:
                sub_segment.parent_id = segment.segment_id
                sub_segment.path = segment.path
                sub_segment.level = segment.level
                refined.append(sub_segment)
        return refined

    def _build_segments(
        self,
        *,
        header: List[str],
        row_chunks,
        delimiter: str,
        metadata: Dict[str, Any],
        table_summary: Optional[str],
    ) -> List[Segment]:
        final_segments: List[Segment] = []
        table_chunk_total = len(row_chunks)

        for table_chunk_index, row_chunk in enumerate(row_chunks):
            original_content = render_csv_table(header, row_chunk.rows, delimiter=delimiter)
            chunk_summary = self._summarize_chunk(header, row_chunk.rows)

            final_content = original_content
            if self.inject_summaries_into_content:
                final_content = build_summary_content(
                    original_content,
                    table_summary=table_summary,
                    chunk_summary=chunk_summary,
                )

            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "is_table": True,
                    "output_format": "csv",
                    "split_strategy": self.__class__.__name__,
                    "chunk_index": table_chunk_index,
                    "chunk_total": table_chunk_total,
                    "table_chunk_index": table_chunk_index,
                    "table_chunk_total": table_chunk_total,
                    "data_row_start": row_chunk.data_row_start,
                    "data_row_end": row_chunk.data_row_end,
                    "data_row_count": len(row_chunk.rows),
                    "delimiter": delimiter,
                    "use_first_row_as_header": self.use_first_row_as_header,
                }
            )

            if table_summary:
                chunk_metadata["table_summary"] = table_summary
            if chunk_summary:
                chunk_metadata["chunk_summary"] = chunk_summary
            if self.inject_summaries_into_content:
                chunk_metadata["original_content"] = original_content

            final_segments.append(
                Segment(
                    content=final_content,
                    type=SegmentType.TABLE,
                    original_format="csv",
                    metadata=chunk_metadata,
                    segment_id=str(uuid.uuid4()),
                )
            )

        return final_segments

    def _summarize_table(self, header: List[str], rows: List[List[str]]) -> Optional[str]:
        if not self.summarizer or not self.summarize_table:
            return None
        markdown_table = render_markdown_table(header, rows)
        if not markdown_table:
            return None
        try:
            return self.summarizer.summarize(markdown_table)
        except Exception:
            return None

    def _summarize_chunk(self, header: List[str], rows: List[List[str]]) -> Optional[str]:
        if not self.summarizer or not self.summarize_chunks:
            return None
        markdown_chunk = render_markdown_table(header, rows)
        if not markdown_chunk:
            return None
        try:
            return self.summarizer.summarize(markdown_chunk)
        except Exception:
            return None
