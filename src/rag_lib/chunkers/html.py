from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
import uuid

from rag_lib.chunkers.base import TextSplitter
from rag_lib.chunkers.table_rows import (
    ParsedTable,
    build_summary_content,
    chunk_table_rows,
    parse_markdown_table,
    render_markdown_table,
)
from rag_lib.config import Settings
from rag_lib.core.domain import Document, Segment, SegmentType
from rag_lib.html_processing import (
    HTMLBlock,
    extract_structural_blocks,
    parse_html_document,
    parse_html_table_element,
    render_html_table,
    strip_non_content_nodes,
)
from rag_lib.summarizers.table import TableSummarizer


@dataclass
class _HeadingContext:
    level: int
    title: str
    path: List[str]
    parent_id: Optional[str]
    segment_id: Optional[str]
    heading_markdown: str
    heading_html: str


class HTMLSplitter(TextSplitter):
    """
    Splits HTML by heading hierarchy while preserving lists/tables/other blocks.
    Strict behavior: parser/render/summarizer errors are propagated to caller.
    """

    def __init__(
        self,
        *,
        output_format: str = "markdown",
        split_table_rows: bool = False,
        use_first_row_as_header: bool = True,
        max_rows_per_chunk: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        summarizer: Optional[TableSummarizer] = None,
        summarize_table: bool = True,
        summarize_chunks: bool = False,
        inject_summaries_into_content: bool = False,
        include_parent_content: Union[bool, int] = False,
    ):
        if output_format not in {"markdown", "html"}:
            raise ValueError("output_format must be 'markdown' or 'html'")

        settings = Settings()
        resolved_rows_per_chunk = (
            settings.ingestion.chunk_size
            if max_rows_per_chunk is None
            else max_rows_per_chunk
        )
        super().__init__(chunk_size=resolved_rows_per_chunk, chunk_overlap=0)

        self.output_format = output_format
        self.split_table_rows = split_table_rows
        self.use_first_row_as_header = use_first_row_as_header
        self.max_rows_per_chunk = resolved_rows_per_chunk
        self.max_chunk_size = max_chunk_size
        self.summarizer = summarizer
        self.summarize_table = summarize_table
        self.summarize_chunks = summarize_chunks
        self.inject_summaries_into_content = inject_summaries_into_content
        self.include_parent_content = include_parent_content
        self._min_concat_level = self._resolve_min_concat_level(include_parent_content)

    def split_text(self, text: str) -> List[str]:
        return [segment.content for segment in self.create_segments(text)]

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        base_metadata = metadata.copy() if metadata else {}
        document = parse_html_document(text)
        strip_non_content_nodes(document)
        blocks = extract_structural_blocks(document)

        segments: List[Segment] = []
        heading_stack: List[_HeadingContext] = []
        current_parts: List[str] = []

        def current_level() -> int:
            return heading_stack[-1].level if heading_stack else 0

        def current_path() -> List[str]:
            return list(heading_stack[-1].path) if heading_stack else []

        def current_parent_id() -> Optional[str]:
            if not heading_stack:
                return None
            context = heading_stack[-1]
            return context.parent_id

        def current_title() -> Optional[str]:
            return heading_stack[-1].title if heading_stack else None

        def flush_text_segment() -> None:
            nonlocal current_parts
            joined = "\n\n".join(part.strip() for part in current_parts if part and part.strip()).strip()
            current_parts = []
            if not joined:
                return

            if self._min_concat_level is not None:
                prefix = self._build_parent_prefix(heading_stack)
                if prefix:
                    joined = f"{prefix}\n\n{joined}"

            segment_metadata = base_metadata.copy()
            segment_metadata["split_strategy"] = self.__class__.__name__
            title = current_title()
            if title:
                segment_metadata["title"] = title

            new_segment = Segment(
                content=joined,
                type=SegmentType.TEXT,
                original_format=self.output_format,
                metadata=segment_metadata,
                segment_id=str(uuid.uuid4()),
                parent_id=current_parent_id(),
                level=current_level(),
                path=current_path(),
            )
            segments.append(new_segment)

            if heading_stack and heading_stack[-1].segment_id is None:
                heading_stack[-1].segment_id = new_segment.segment_id

        for block in blocks:
            if block.kind == "heading":
                flush_text_segment()
                heading_level = block.heading_level
                heading_title = block.heading_title
                if heading_level is None or heading_title is None:
                    raise ValueError("Heading block is missing level/title.")

                while heading_stack and heading_stack[-1].level >= heading_level:
                    heading_stack.pop()

                parent_id = heading_stack[-1].segment_id if heading_stack else None
                path = [entry.title for entry in heading_stack]
                heading_stack.append(
                    _HeadingContext(
                        level=heading_level,
                        title=heading_title,
                        path=path,
                        parent_id=parent_id,
                        segment_id=None,
                        heading_markdown=block.markdown,
                        heading_html=block.html,
                    )
                )
                current_parts = [self._render_block(block)]
                continue

            if block.kind == "table":
                flush_text_segment()
                if block.parsed_table is None:
                    raise ValueError("Table block is missing parsed table data.")

                table_segments = self._emit_table_segments(
                    parsed_table=block.parsed_table,
                    metadata=base_metadata,
                    context=heading_stack[-1] if heading_stack else None,
                    original_markdown=block.markdown,
                    original_html=block.html,
                )
                segments.extend(table_segments)
                if heading_stack and heading_stack[-1].segment_id is None and table_segments:
                    heading_stack[-1].segment_id = table_segments[0].segment_id
                continue

            current_parts.append(self._render_block(block))

        flush_text_segment()
        self._update_chunk_metadata(segments)
        return segments

    def split_documents(self, documents: Iterable[Document]) -> List[Segment]:
        all_segments: List[Segment] = []
        for document in documents:
            all_segments.extend(self.create_segments(document.page_content, metadata=document.metadata))
        return all_segments

    def split_segments(self, segments: Iterable[Segment]) -> List[Segment]:
        refined: List[Segment] = []
        process_table_segments = (
            self.split_table_rows
            or not self.use_first_row_as_header
            or self.inject_summaries_into_content
            or (self.summarizer is not None and (self.summarize_table or self.summarize_chunks))
        )

        for segment in segments:
            if segment.type != SegmentType.TABLE or not process_table_segments:
                refined.append(segment)
                continue

            parsed_table, original_markdown, original_html = self._parse_table_segment_content(segment)
            sub_segments = self._emit_table_segments(
                parsed_table=parsed_table,
                metadata=segment.metadata,
                context=None,
                original_markdown=original_markdown,
                original_html=original_html,
            )
            for sub_segment in sub_segments:
                sub_segment.parent_id = segment.segment_id
                sub_segment.path = segment.path
                sub_segment.level = segment.level
                refined.append(sub_segment)

        self._update_chunk_metadata(refined)
        return refined

    def _emit_table_segments(
        self,
        *,
        parsed_table: ParsedTable,
        metadata: Dict[str, Any],
        context: Optional[_HeadingContext],
        original_markdown: Optional[str],
        original_html: Optional[str],
    ) -> List[Segment]:
        if not parsed_table.header:
            raise ValueError("Parsed table header is empty.")

        header = parsed_table.header
        rows = parsed_table.rows
        markdown_table_full = render_markdown_table(header, rows)

        level = context.level if context else 0
        path = list(context.path) if context else []
        parent_id = (context.segment_id or context.parent_id) if context else None
        title = context.title if context else None

        if self.split_table_rows:
            if not rows:
                return []

            table_summary = self._summarize_markdown(markdown_table_full, summary_level="table")
            row_chunks = chunk_table_rows(
                header=header,
                rows=rows,
                render_chunk=self._render_table_output,
                max_rows_per_chunk=self.max_rows_per_chunk,
                max_chunk_size=self.max_chunk_size,
                length_function=self._length_function,
            )

            total_chunks = len(row_chunks)
            output_segments: List[Segment] = []
            for table_chunk_index, row_chunk in enumerate(row_chunks):
                original_content = self._render_table_output(header, row_chunk.rows)
                chunk_markdown = render_markdown_table(header, row_chunk.rows)
                chunk_summary = self._summarize_markdown(chunk_markdown, summary_level="chunk")

                final_content = original_content
                if self.inject_summaries_into_content:
                    final_content = build_summary_content(
                        original_content,
                        table_summary=table_summary,
                        chunk_summary=chunk_summary,
                    )

                table_metadata = metadata.copy()
                table_metadata.update(
                    {
                        "is_table": True,
                        "output_format": self.output_format,
                        "split_strategy": self.__class__.__name__,
                        "table_chunk_index": table_chunk_index,
                        "table_chunk_total": total_chunks,
                        "data_row_start": row_chunk.data_row_start,
                        "data_row_end": row_chunk.data_row_end,
                        "data_row_count": len(row_chunk.rows),
                        "use_first_row_as_header": self.use_first_row_as_header,
                    }
                )
                if title:
                    table_metadata["title"] = title
                if table_summary:
                    table_metadata["table_summary"] = table_summary
                if chunk_summary:
                    table_metadata["chunk_summary"] = chunk_summary
                if self.inject_summaries_into_content:
                    table_metadata["original_content"] = original_content

                output_segments.append(
                    Segment(
                        content=final_content,
                        type=SegmentType.TABLE,
                        original_format=self.output_format,
                        metadata=table_metadata,
                        segment_id=str(uuid.uuid4()),
                        parent_id=parent_id,
                        level=level,
                        path=path,
                    )
                )

            return output_segments

        use_original = self.use_first_row_as_header
        table_content_for_output = self._render_table_output(
            header,
            rows,
            original_markdown=original_markdown,
            original_html=original_html,
            use_original=use_original,
        )
        table_summary = self._summarize_markdown(markdown_table_full, summary_level="table")
        chunk_summary = self._summarize_markdown(markdown_table_full, summary_level="chunk")

        final_content = table_content_for_output
        if self.inject_summaries_into_content:
            final_content = build_summary_content(
                table_content_for_output,
                table_summary=table_summary,
                chunk_summary=chunk_summary,
            )

        table_metadata = metadata.copy()
        table_metadata.update(
            {
                "is_table": True,
                "output_format": self.output_format,
                "split_strategy": self.__class__.__name__,
                "table_chunk_index": 0,
                "table_chunk_total": 1,
                "use_first_row_as_header": self.use_first_row_as_header,
                "data_row_start": 0,
                "data_row_end": len(rows),
                "data_row_count": len(rows),
            }
        )
        if title:
            table_metadata["title"] = title
        if table_summary:
            table_metadata["table_summary"] = table_summary
        if chunk_summary:
            table_metadata["chunk_summary"] = chunk_summary
        if self.inject_summaries_into_content:
            table_metadata["original_content"] = table_content_for_output

        return [
            Segment(
                content=final_content,
                type=SegmentType.TABLE,
                original_format=self.output_format,
                metadata=table_metadata,
                segment_id=str(uuid.uuid4()),
                parent_id=parent_id,
                level=level,
                path=path,
            )
        ]

    def _parse_table_segment_content(
        self,
        segment: Segment,
    ) -> tuple[ParsedTable, Optional[str], Optional[str]]:
        source_format = (segment.original_format or segment.metadata.get("output_format", "")).lower()

        if source_format == "markdown":
            parsed = parse_markdown_table(
                segment.content,
                use_first_row_as_header=self.use_first_row_as_header,
            )
            if not parsed.header:
                raise ValueError("Invalid markdown table segment content.")
            return parsed, segment.content, None

        if source_format == "html":
            document = parse_html_document(segment.content)
            strip_non_content_nodes(document)
            table_nodes = document.xpath(".//table")
            if len(table_nodes) != 1:
                raise ValueError("HTML table segment must contain exactly one table element.")
            parsed = parse_html_table_element(
                table_nodes[0],
                use_first_row_as_header=self.use_first_row_as_header,
            )
            return parsed, None, segment.content

        raise ValueError(f"Unsupported table segment format: {source_format!r}")

    def _render_block(self, block: HTMLBlock) -> str:
        return block.markdown if self.output_format == "markdown" else block.html

    def _render_table_output(
        self,
        header: List[str],
        rows: List[List[str]],
        *,
        original_markdown: Optional[str] = None,
        original_html: Optional[str] = None,
        use_original: bool = False,
    ) -> str:
        if self.output_format == "markdown":
            if use_original and original_markdown:
                return original_markdown
            return render_markdown_table(header, rows)

        if use_original and original_html:
            return original_html
        return render_html_table(header, rows)

    def _summarize_markdown(self, markdown_table: str, summary_level: str) -> Optional[str]:
        if not self.summarizer:
            return None
        if summary_level == "table" and not self.summarize_table:
            return None
        if summary_level == "chunk" and not self.summarize_chunks:
            return None
        if not markdown_table.strip():
            return None
        return self.summarizer.summarize(markdown_table)

    def _build_parent_prefix(self, heading_stack: List[_HeadingContext]) -> str:
        if self._min_concat_level is None:
            return ""
        if len(heading_stack) <= 1:
            return ""

        parents = heading_stack[:-1]
        pieces: List[str] = []
        for parent in parents:
            if parent.level < self._min_concat_level:
                continue
            pieces.append(
                parent.heading_markdown if self.output_format == "markdown" else parent.heading_html
            )
        return "\n\n".join(piece.strip() for piece in pieces if piece.strip()).strip()

    def _resolve_min_concat_level(self, include_parent_content: Union[bool, int]) -> Optional[int]:
        if isinstance(include_parent_content, bool):
            return 0 if include_parent_content else None
        return include_parent_content

    def _update_chunk_metadata(self, segments: List[Segment]) -> None:
        total = len(segments)
        for index, segment in enumerate(segments):
            segment.metadata["chunk_index"] = index
            segment.metadata["chunk_total"] = total
