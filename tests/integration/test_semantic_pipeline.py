import pytest
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.embeddings.mock import MockEmbeddings

def test_semantic_pipeline_e2e():
    """
    Demonstrates the semantic chunking pipeline on a larger text.
    """
    # 1. Setup components
    embeddings = MockEmbeddings()
    # Lower threshold to ensure we get some groupings but also some splits
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.5)
    
    # 2. Input Data (Simulating a document with multiple distinct sections)
    # Using keywords our MockEmbeddings knows: "apple", "banana", "dog", "cat", "space", "planet"
    text = (
        "Introduction to Fruits. "
        "I really like apple. Apple is a healthy fruit. "
        "Banana is also a fruit. It is yellow. "
        "Fruits are good for you. "
        # Shift to Space
        "Now let's talk about space. "
        "The planet Mars is red. Space is vast. "
        "Astronauts travel to space. "
        # Shift to Pets
        "Back on Earth, I have a dog. "
        "My dog likes to chase the cat. "
        "The cat climbs the tree. "
    )
    
    # 3. Execution
    segments = chunker.chunk(text)
    
    # 4. Verification
    print(f"\n[PIPELINE] Input Text Length: {len(text)} chars")
    print(f"[PIPELINE] Generated {len(segments)} Semantic Segments:")
    
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: '{seg.content}'")
        
    # We expect roughly 3 segments (Fruit, Space, Pets)
    # Note: "Introduction to Fruits." might be its own or grouped.
    # "Now let's talk about space" might bridge or start new.
    
    assert len(segments) >= 3, "Should detect at least 3 distinct topics"
    
    # Check content of segments to ensure they aren't mixed wildly
    # Segment 1 should have fruit terms
    assert "apple" in segments[0].content.lower() or "fruit" in segments[0].content.lower()
    
    # Last segment should have pet terms
    assert "dog" in segments[-1].content.lower() or "cat" in segments[-1].content.lower()

if __name__ == "__main__":
    test_semantic_pipeline_e2e()
