from abc import ABC, abstractmethod
from typing import List, Callable, Iterable
import uuid
from rag_lib.core.domain import Segment

class TextSplitter(ABC):
    """
    Abstract Base Class for text splitting strategies.
    Aligns with LangChain's interface but operates on Domain Segments.
    """
    def __init__(
        self, 
        chunk_size: int = 4000, 
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Core splitting logic. Must be implemented by subclasses.
        Accepts a raw string and returns a list of chunk strings.
        """
        pass

    def split_segments(self, segments: Iterable[Segment]) -> List[Segment]:
        """
        Splits a list of Segments into smaller chunks, strictly preserving metadata.
        """
        final_segments = []
        for segment in segments:
            chunks = self.split_text(segment.content)
            search_start_idx = 0
            
            for i, chunk_content in enumerate(chunks):
                new_meta = segment.metadata.copy()
                
                # Calculate start_index (best effort)
                start_index = segment.content.find(chunk_content, search_start_idx)
                if start_index != -1:
                    # Advance search position to avoid finding the same substring context again
                    # This assumes chunks appear in order in the original text (which they should)
                    search_start_idx = start_index + len(chunk_content)
                
                # --- Defined Metadata Schema ---
                new_meta.update({
                    "parent_id": segment.segment_id,   # Link to source Segment
                    "chunk_index": i,                  # Order in sequence
                    "chunk_total": len(chunks),        # Total chunks from this segment
                    "start_index": start_index,        # Offset in original text
                    "split_strategy": self.__class__.__name__
                })
                # -------------------------------

                new_seg = Segment(
                    content=chunk_content,
                    path=segment.path,       # Inherit path
                    level=segment.level,     # Inherit level
                    metadata=new_meta,
                    segment_id=str(uuid.uuid4())
                )
                final_segments.append(new_seg)
        return final_segments
