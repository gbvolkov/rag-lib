import pytest
import uuid
from typing import List
from rag_lib.core.domain import Segment
from rag_lib.chunkers.base import TextSplitter

# --- Mock Implementation for Testing ---
class MockSplitter(TextSplitter):
    """A simple splitter that splits by space for testing the base contract."""
    def split_text(self, text: str) -> List[str]:
        return text.split(" ")

# --- Tests ---

def test_base_splitter_metadata_contract():
    """
    T-01: Verify that split_segments correctly preserves metadata 
    and generates the required lineage fields.
    """
    splitter = MockSplitter()
    
    # Input Segment
    original_id = str(uuid.uuid4())
    input_seg = Segment(
        content="Alpha Beta Gamma",
        path=["Section 1"],
        level=2,
        metadata={"source": "doc.txt", "original_key": "value"}
    )
    # Manually set ID to verify parent linking (if Segment allows setting it, 
    # otherwise we capture the auto-generated one)
    # Assuming Segment generates ID on init, we'll use input_seg.segment_id
    
    # Execute
    try:
        results = splitter.split_segments([input_seg])
    except NotImplementedError:
        pytest.fail("split_segments is not implemented yet")

    # Verification
    assert len(results) == 3
    
    # Check First Chunk
    c0 = results[0]
    assert c0.content == "Alpha"
    assert c0.path == ["Section 1"]
    assert c0.level == 2 # Should match parent
    assert c0.metadata["parent_id"] == input_seg.segment_id
    assert c0.metadata["chunk_index"] == 0
    assert c0.metadata["chunk_total"] == 3
    assert c0.metadata["split_strategy"] == "MockSplitter"
    assert c0.metadata["source"] == "doc.txt"
    assert c0.metadata["original_key"] == "value"
    assert c0.metadata["start_index"] == 0

    # Check Last Chunk
    c2 = results[2]
    assert c2.content == "Gamma"
    assert c2.metadata["chunk_index"] == 2
