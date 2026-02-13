import pytest
from pydantic import ValidationError
from rag_lib.core.domain import Segment, SegmentType

def test_segment_defaults():
    """Verify default segment is TEXT type."""
    seg = Segment(content="Hello", segment_id="123")
    assert seg.type == SegmentType.TEXT
    assert seg.original_format == "text"

def test_segment_table_type():
    """Verify we can create a TABLE segment."""
    md_table = "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |"
    seg = Segment(
        content=md_table,
        segment_id="123",
        type=SegmentType.TABLE,
        original_format="markdown"
    )
    assert seg.type == SegmentType.TABLE
    assert seg.original_format == "markdown"
    assert seg.content == md_table

def test_type_validation():
    """Verify strictly typed enum."""
    # Pydantic should allow string coercion if it matches the Enum value
    seg = Segment(content="test", segment_id="1", type="table")
    assert seg.type == SegmentType.TABLE
    
    # But invalid strings should fail (depending on config, usually ValidationError)
    with pytest.raises(ValidationError):
        Segment(content="test", segment_id="1", type="INVALID_TYPE")
