from typing import List, Optional, Any, Dict
import uuid
from rag_lib.chunkers.base import TextSplitter
from rag_lib.core.domain import Segment, SegmentType

class QASplitter(TextSplitter):
    """
    Splits text containing Question-Answer pairs (prefixed with 'Q: ').
    Extracts the question into metadata.
    """
    def __init__(self):
        super().__init__()

    def split_text(self, text: str) -> List[str]:
        """
        Splits text by 'Q: '. Reconstructs full block.
        """
        parts = text.split("Q: ")
        chunks = []
        for part in parts:
            if not part.strip():
                continue
            # Reconstruct "Q: " prefix if it was split
            # Note: split removes delimiter.
            # We assume user wants the full text block.
            chunk = "Q: " + part
            chunks.append(chunk)
        return chunks

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        if metadata is None:
            metadata = {}
            
        chunks = self.split_text(text)
        segments = []
        
        for chunk in chunks:
            # Extract question for metadata
            # Assume question is first line or until 'A:'?
            # Simple heuristic: First line after Q:
            try:
                # chunk starts with "Q: "
                # find end of question
                question_end = chunk.find("\n")
                if question_end == -1:
                    question_end = len(chunk)
                
                # "Q: This is the question" -> "This is the question"
                question = chunk[3:question_end].strip()
            except Exception:
                question = "Unknown"

            meta = metadata.copy()
            meta["question"] = question
            meta["type"] = "qa"

            segments.append(Segment(
                content=chunk,
                segment_id=str(uuid.uuid4()),
                type=SegmentType.TEXT,
                metadata=meta
            ))
            
        return segments
