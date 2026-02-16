import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from rag_lib.core.domain import Segment, SegmentType

try:
    print("Attempting to create Segment...")
    s = Segment(content="test content", original_format="markdown", segment_id="md_0")
    print(f"Success: {s}")
except Exception as e:
    print(f"Error: {e}")
