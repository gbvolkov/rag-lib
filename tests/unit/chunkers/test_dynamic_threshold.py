import pytest
import numpy as np
from typing import List
from unittest.mock import MagicMock
from langchain_core.embeddings import Embeddings
from rag_lib.chunkers.semantic import SemanticChunker

class MockVectorEmbeddings(Embeddings):
    """
    Mock embeddings that return pre-defined vectors based on input text.
    Allows us to control exact cosine similarities.
    """
    def __init__(self, vector_map: dict):
        self.vector_map = vector_map

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Return mapped vector or zero vector
        return [self.vector_map.get(t, [0.0, 0.0]) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.vector_map.get(text, [0.0, 0.0])

def test_dynamic_threshold_percentile():
    # Setup vectors:
    # A and B are identical (sim=1.0)
    # B and C are orthogonal (sim=0.0)
    # C and D are identical (sim=1.0)
    # D and E are somewhat similar (sim=0.5)
    # Sequence: A, B, C, D, E
    # Pairs: (A,B)=1.0, (B,C)=0.0, (C,D)=1.0, (D,E)=0.5
    # Similarities: [1.0, 0.0, 1.0, 0.5]
    
    vec_map = {
        "A": [1.0, 0.0],
        "B": [1.0, 0.0],
        "C": [0.0, 1.0],
        "D": [0.0, 1.0],
        "E": [0.0, 0.8] # 0.8 is roughly sim 0.8 with [0,1]? No.
                        # sim([0,1], [0, 0.8]) -> dot=0.8 / (1*0.8) = 1.0 same direction.
                        # Let's use [0.707, 0.707] for 45 deg = 0.707 sim with [1,0]
    }
    # Redefine for clarity
    # Sim(v1, v2) = dot(v1, v2) (assume normalized)
    v1 = [1.0, 0.0] 
    v2 = [0.0, 1.0] # sim 0
    v3 = [0.7071, 0.7071] # sim 0.707 with v1 and v2
    
    # Let's construct a scenario:
    # 5 sentences. 
    # S1-S2: High Sim (0.9)
    # S2-S3: Low Sim (0.2) -> Should split here
    # S3-S4: High Sim (0.9)
    # S4-S5: Medium Sim (0.5) -> Maybe split?
    
    # Similarities list: [0.9, 0.2, 0.9, 0.5]
    # Sorted: [0.2, 0.5, 0.9, 0.9]
    
    # Case 1: Percentile 30
    # 30th percentile of [0.2, 0.5, 0.9, 0.9]
    # np.percentile -> roughly 0.4 something.
    # So Threshold ~0.4.
    # Splits: 
    # 0.9 >= 0.4 (Join)
    # 0.2 < 0.4 (SPLIT)
    # 0.9 >= 0.4 (Join)
    # 0.5 >= 0.4 (Join)
    # Result: 2 chunks (S1-S2), (S3-S5)
    
    # Case 2: Percentile 60
    # 60th percentile of [0.2, 0.5, 0.9, 0.9]
    # Roughly 0.9. (Actually between 0.5 and 0.9)
    # Let's say Threshold ~0.7
    # Splits:
    # 0.9 >= 0.7 (Join)
    # 0.2 < 0.7 (SPLIT)
    # 0.9 >= 0.7 (Join)
    # 0.5 < 0.7 (SPLIT)
    # Result: 3 chunks (S1-S2), (S3-S4), (S5)
    
    # Verify this logic implementation
    
    # Construct exact text/vectors for S1..S5 isn't strictly needed if we mock _cosine_similarity directly!
    # That's easier/safer than vector math in test setup.
    
    mock_embeddings = MagicMock(spec=Embeddings)
    # We don't care about embed_documents output, we will mock _cosine_similarity via side_effect or subclass
    mock_embeddings.embed_documents.return_value = [[0]] * 5 # dummy vectors
    
    # But wait, SemanticChunker calls self._cosine_similarity(vectors[i], vectors[j])
    # Hard to mock a method on the class under test effectively if we instantiate it normally.
    # Better to subclass SemanticChunker in test for control.
    
class TestableSemanticChunker(SemanticChunker):
    def __init__(self, predefined_similarities: List[float], **kwargs):
        # We bypass embeddings init partially
        self.predefined_similarities = predefined_similarities
        self.sim_index = 0
        super().__init__(embeddings=MagicMock(), **kwargs)
        
    def _cosine_similarity(self, v1, v2) -> float:
        # Return next predefined sim
        val = self.predefined_similarities[self.sim_index]
        self.sim_index += 1
        return val

def test_chunking_logic_percentile_low():
    # Similarities: [0.9, 0.2, 0.9, 0.5]
    # With P30, threshold should be around 0.4.
    # Expect split at 0.2 only.
    
    sims = [0.9, 0.2, 0.9, 0.5]
    chunker = TestableSemanticChunker(
        predefined_similarities=sims,
        threshold_type="percentile",
        percentile_threshold=30
    )
    
    text = "S1. S2. S3. S4. S5." # 5 sentences -> 4 intervals
    segments = chunker.chunk(text)
    
    # Expect splits where sim < threshold.
    # threshold = np.percentile([0.2, 0.5, 0.9, 0.9], 30) -> 0.2 + (0.5-0.2)*0.9 = 0.47
    # 0.9 > 0.47 -> Join (S1, S2)
    # 0.2 < 0.47 -> Split! End (S1, S2), Start (S3)
    # 0.9 > 0.47 -> Join (S3, S4)
    # 0.5 > 0.47 -> Join (S3, S4, S5)
    
    # So 2 segments.
    # Seg1: S1 S2
    # Seg2: S3 S4 S5
    
    assert len(segments) == 2
    assert "S1. S2." in segments[0].content
    assert "S3. S4. S5." in segments[1].content

def test_chunking_logic_percentile_high():
    # Similarities: [0.9, 0.2, 0.9, 0.5]
    # With P60
    # threshold = np.percentile([0.2, 0.5, 0.9, 0.9], 60) -> 0.5 + (0.9-0.5)*0.8 = 0.82
    
    sims = [0.9, 0.2, 0.9, 0.5]
    chunker = TestableSemanticChunker(
        predefined_similarities=sims,
        threshold_type="percentile",
        percentile_threshold=60
    )
    
    text = "S1. S2. S3. S4. S5."
    segments = chunker.chunk(text)
    
    # 0.9 > 0.82 -> Join
    # 0.2 < 0.82 -> Split
    # 0.9 > 0.82 -> Join
    # 0.5 < 0.82 -> Split
    
    # Segments: (S1, S2), (S3, S4), (S5)
    assert len(segments) == 3
    assert "S1. S2." in segments[0].content
    assert "S3. S4." in segments[1].content
    assert "S5." in segments[2].content

def test_fallback_fixed():
    sims = [0.9, 0.2, 0.9, 0.5]
    # Fixed threshold 0.6
    chunker = TestableSemanticChunker(
        predefined_similarities=sims,
        threshold_type="fixed",
        threshold=0.6
    )
    text = "S1. S2. S3. S4. S5."
    segments = chunker.chunk(text)
    
    # 0.9 >= 0.6 -> Join
    # 0.2 < 0.6 -> Split
    # 0.9 >= 0.6 -> Join
    # 0.5 < 0.6 -> Split
    
    assert len(segments) == 3
