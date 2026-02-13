from rag_lib.core.domain import Segment, SegmentType

def test_segment_types():
    # Verify new types exist
    assert SegmentType.IMAGE == "image"
    assert SegmentType.AUDIO == "audio"
    
    # Verify instantiation
    seg = Segment(content="image.png", type=SegmentType.IMAGE)
    assert seg.type == SegmentType.IMAGE
    assert seg.type.value == "image"
