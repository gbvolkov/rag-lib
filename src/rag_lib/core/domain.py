from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from enum import Enum

class SegmentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    OTHER = "other"

class Segment(BaseModel):
    """
    A logical part of a document (e.g., Chapter, Section, Row) that acts as a standalone unit.
    Wraps LangChain's Document concept but adds explicit hierarchy tracking.
    """
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Hierarchy / Lineage
    segment_id: Optional[str] = None
    parent_id: Optional[str] = None
    level: int = 0  # 0=Root, 1=H1, 2=H2...
    path: List[str] = Field(default_factory=list) # e.g. ["Chapter 1", "Section 1.1"]
    
    # Type / Format
    type: SegmentType = SegmentType.TEXT
    original_format: str = "text" # "text", "markdown", "csv", etc.

    def to_langchain(self) -> Document:
        """
        Convert to a LangChain Document.
        Merging hierarchy fields into metadata for compatibility.
        """
        meta = self.metadata.copy()
        if self.segment_id:
            meta["segment_id"] = self.segment_id
        if self.parent_id:
            meta["parent_id"] = self.parent_id
        meta["level"] = self.level
        meta["path"] = self.path
        meta["type"] = self.type.value
        meta["original_format"] = self.original_format
        
        return Document(page_content=self.content, metadata=meta)
