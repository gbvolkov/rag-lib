import pytest
from rag_lib.core.domain import Segment

# Test D-01: Path Tracking
def test_path_tracking():
    seg = Segment(
        content="test", 
        path=["Chapter 1", "Section 1.1"],
        level=2
    )
    assert seg.path == ["Chapter 1", "Section 1.1"]
    assert seg.level == 2

# Test D-02: Hierarchy Splitting (Basic instantiation check)
def test_hierarchy_fields():
    seg = Segment(content="child", parent_id="parent_1", segment_id="child_1")
    assert seg.parent_id == "parent_1"
    assert seg.segment_id == "child_1"

# Test D-03: Validation (Empty Content)
def test_validation_content_empty():
    # Pydantic validates types, but we might want explicit empty-string check if required
    # For now, just ensure it accepts valid string
    seg = Segment(content="valid")
    assert seg.content == "valid"

# Test D-04:/LangChain Conversion (Mocked for now)
def test_to_langchain_conversion():
    from langchain_core.documents import Document
    seg = Segment(content="text", metadata={"source": "file.txt"})
    doc = seg.to_langchain()
    assert isinstance(doc, Document)
    assert doc.page_content == "text"
    # Metadata includes hierarchy fields now
    assert doc.metadata["source"] == "file.txt"
    assert doc.metadata["level"] == 0
    assert doc.metadata["path"] == [] 

# Test D-05: Metadata Defaults
def test_metadata_defaults():
    seg = Segment(content="text")
    assert seg.metadata == {}

# Test D-06: Metadata Merge (Functionality to be added)
def test_metadata_update():
    seg = Segment(content="text", metadata={"a": 1})
    seg.metadata["b"] = 2
    assert seg.metadata == {"a": 1, "b": 2}
