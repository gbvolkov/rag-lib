from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class GraphNode(BaseModel):
    """
    Represents a node in the Knowledge Graph.
    Can be an Entity (Person, Location) or a Segment reference.
    """
    id: str  # Unique ID (e.g., "Entity:Albert Einstein" or "Segment:uuid")
    type: str  # "entity", "segment", "chunk"
    label: str  # Display label
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_segment_id: Optional[str] = None  # Traceability

class GraphEdge(BaseModel):
    """
    Represents a relationship between two nodes.
    """
    source_id: str
    target_id: str
    relation_type: str  # "WORKS_FOR", "LOCATED_IN", "MENTIONS"
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_segment_id: Optional[str] = None # Traceability
