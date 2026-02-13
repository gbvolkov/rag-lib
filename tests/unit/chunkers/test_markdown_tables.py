import pytest
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter

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
    segments = splitter.split_text(content)
    
    # Expect 3 segments: Text, Table, Text
    assert len(segments) == 3
    
    assert segments[0].type == SegmentType.TEXT
    assert "Here is a price list" in segments[0].content
    
    assert segments[1].type == SegmentType.TABLE
    assert "| Item | Price |" in segments[1].content
    assert "| Apple | $1 |" in segments[1].content
    assert segments[1].original_format == "markdown"
    
    assert segments[2].type == SegmentType.TEXT
    assert "End of list" in segments[2].content

def test_no_table():
    """Cycle 3.3: Pass through text without tables."""
    content = "Just text.\nNo tables here."
    splitter = MarkdownTableSplitter()
    segments = splitter.split_text(content)
    
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
    segments = splitter.split_text(content)
    
    table_segs = [s for s in segments if s.type == SegmentType.TABLE]
    assert len(table_segs) == 2
    assert "| A | B |" in table_segs[0].content
    assert "| X | Y |" in table_segs[1].content
