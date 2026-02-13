import pytest
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.embeddings.mock import MockEmbeddings
from rag_lib.core.domain import SegmentType

def test_semantic_chunking_topic_shift():
    # Setup Mock Embeddings with knwon concepts
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.6) # Threshold is 1 - distance, or similarity directly? Usually similarity.
    # If using similarity: High means same topic. Break if < 0.6.
    
    # Text with two distinct topics: Fruit and Pets
    # "apple" and "banana" -> High Similarity
    # "dog" and "cat" -> High Similarity
    # "banana" vs "dog" -> Low Similarity
    text = (
        "I like to eat apple. "
        "Apple is a tasty fruit. "
        "Banana is yellow and sweet. "  # End of Topic 1
        "I have a pet dog. "            # Start of Topic 2 (Break expected here)
        "The dog barks at the cat. "
        "Cat loves to sleep."
    )
    
    segments = chunker.chunk(text)
    
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    
    # Check contents
    assert "apple" in segments[0].content.lower()
    assert "banana" in segments[0].content.lower()
    assert "dog" not in segments[0].content.lower()
    
    assert "dog" in segments[1].content.lower()
    assert "cat" in segments[1].content.lower()
    assert "apple" not in segments[1].content.lower()
    
    assert segments[0].type == SegmentType.TEXT

def test_semantic_chunking_single_topic():
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.5)
    
    text = "Apple is red. I like apple. Apple pie is good."
    segments = chunker.chunk(text)
    
    assert len(segments) == 1

