from rag_lib.core.domain import SegmentType
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.summarizers.table import TableSummarizer


class MockSummarizer(TableSummarizer):
    def summarize(self, markdown_table: str) -> str:
        return f"summary(len={len(markdown_table)})"

def test_extract_markdown_table_from_text():
    """Cycle 3.3: Detect a markdown table block inside text."""
    content = """
    Here is a price list:
    
    | Item | Price |
    |---|---|
    | Apple | $1 |
    | Banana | $2 |
    
    End of list.
    """
    
    splitter = MarkdownTableSplitter()
    segments = splitter.create_segments(content)
    
    # Expect 3 segments: Text, Table, Text
    assert len(segments) == 3
    
    assert segments[0].type == SegmentType.TEXT
    assert "Here is a price list" in segments[0].content
    
    assert segments[1].type == SegmentType.TABLE
    assert "| Item | Price |" in segments[1].content
    assert "| Apple | $1 |" in segments[1].content
    assert segments[1].original_format == "markdown"
    assert segments[1].metadata["output_format"] == "markdown"
    
    assert segments[2].type == SegmentType.TEXT
    assert "End of list" in segments[2].content

def test_no_table():
    """Cycle 3.3: Pass through text without tables."""
    content = "Just text.\nNo tables here."
    splitter = MarkdownTableSplitter()
    segments = splitter.create_segments(content)
    
    assert len(segments) == 1
    assert segments[0].type == SegmentType.TEXT
    assert segments[0].content == content

def test_multiple_tables():
    """Cycle 3.3: Handle multiple tables."""
    content = """
    T1:
    | A | B |
    |---|---|
    | 1 | 2 |
    
    Middle text.
    
    T2:
    | X | Y |
    |---|---|
    | 8 | 9 |
    """
    
    splitter = MarkdownTableSplitter()
    segments = splitter.create_segments(content)
    
    table_segs = [s for s in segments if s.type == SegmentType.TABLE]
    assert len(table_segs) == 2
    assert "| A | B |" in table_segs[0].content
    assert "| X | Y |" in table_segs[1].content


def test_table_summary_added_without_row_splitting():
    content = """
    | Item | Price |
    |---|---|
    | Apple | 1 |
    | Banana | 2 |
    """
    splitter = MarkdownTableSplitter(
        summarizer=MockSummarizer(),
        summarize_table=True,
        summarize_chunks=False,
    )

    segments = splitter.create_segments(content)
    table_seg = [s for s in segments if s.type == SegmentType.TABLE][0]

    assert "table_summary" in table_seg.metadata
    assert "chunk_summary" not in table_seg.metadata


def test_row_split_mode_with_table_and_chunk_summaries():
    content = """
    | ID | Value |
    |---|---|
    | 1 | A |
    | 2 | B |
    | 3 | C |
    """
    splitter = MarkdownTableSplitter(
        split_table_rows=True,
        max_rows_per_chunk=2,
        summarizer=MockSummarizer(),
        summarize_table=True,
        summarize_chunks=True,
    )

    segments = splitter.create_segments(content)
    table_segs = [s for s in segments if s.type == SegmentType.TABLE]

    assert len(table_segs) == 2
    assert table_segs[0].metadata["table_chunk_total"] == 2
    assert table_segs[0].metadata["table_summary"] == table_segs[1].metadata["table_summary"]
    assert "chunk_summary" in table_segs[0].metadata
    assert table_segs[0].metadata["data_row_start"] == 0
    assert table_segs[1].metadata["data_row_start"] == 2


def test_row_split_mode_with_summary_injection():
    content = """
    | ID | Value |
    |---|---|
    | 1 | A |
    | 2 | B |
    """
    splitter = MarkdownTableSplitter(
        split_table_rows=True,
        max_rows_per_chunk=1,
        summarizer=MockSummarizer(),
        summarize_table=True,
        summarize_chunks=True,
        inject_summaries_into_content=True,
    )

    segments = splitter.create_segments(content)
    table_seg = [s for s in segments if s.type == SegmentType.TABLE][0]

    assert table_seg.content.startswith("Table Summary:\n")
    assert "\n\n---\n\n" in table_seg.content
    assert "original_content" in table_seg.metadata


def test_row_split_mode_keeps_text_around_tables():
    content = """
    Intro text.

    | ID | Value |
    |---|---|
    | 1 | A |
    | 2 | B |
    | 3 | C |

    Outro text.
    """
    splitter = MarkdownTableSplitter(
        split_table_rows=True,
        max_rows_per_chunk=2,
    )

    segments = splitter.create_segments(content)

    assert segments[0].type == SegmentType.TEXT
    assert "Intro text." in segments[0].content
    assert segments[-1].type == SegmentType.TEXT
    assert "Outro text." in segments[-1].content
    table_segs = [s for s in segments if s.type == SegmentType.TABLE]
    assert len(table_segs) == 2


def test_markdown_table_splitter_can_generate_column_headers_without_row_split():
    content = """
    | ID | Value |
    |---|---|
    | 1 | A |
    | 2 | B |
    """
    splitter = MarkdownTableSplitter(use_first_row_as_header=False)

    segments = splitter.create_segments(content)
    table_seg = [s for s in segments if s.type == SegmentType.TABLE][0]

    assert "| Column1 | Column2 |" in table_seg.content
    assert "| ID | Value |" in table_seg.content
    assert table_seg.metadata["use_first_row_as_header"] is False


def test_markdown_table_splitter_can_generate_column_headers_with_row_split():
    content = """
    | ID | Value |
    |---|---|
    | 1 | A |
    | 2 | B |
    """
    splitter = MarkdownTableSplitter(
        split_table_rows=True,
        max_rows_per_chunk=2,
        use_first_row_as_header=False,
    )

    segments = splitter.create_segments(content)
    table_segs = [s for s in segments if s.type == SegmentType.TABLE]

    assert len(table_segs) == 2
    assert "| Column1 | Column2 |" in table_segs[0].content
    assert "| ID | Value |" in table_segs[0].content
    assert table_segs[0].metadata["data_row_count"] == 2
    assert table_segs[1].metadata["data_row_count"] == 1

