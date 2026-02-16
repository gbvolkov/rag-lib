from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.core.domain import Document, Segment, SegmentType
from langchain_core.embeddings import FakeEmbeddings # Lightweight matching interface

def test_markdown_splitter():
    print("Testing MarkdownTableSplitter...")
    text = """
Before table.

| Col1 | Col2 |
|---|---|
| Val1 | Val2 |

After table.
    """
    doc = Document(page_content=text, metadata={"source": "test_md"})
    
    splitter = MarkdownTableSplitter()
    
    # 1. Document -> Segments
    segments = splitter.split_documents([doc])
    print(f"Segments produced: {len(segments)}")
    for s in segments:
        print(f" - {s.type}: {s.content[:20].replace('\\n', ' ')}...")
        
    assert len(segments) == 3
    assert segments[0].type == SegmentType.TEXT
    assert segments[1].type == SegmentType.TABLE
    assert segments[2].type == SegmentType.TEXT
    
    # 2. Segment -> Chunks (Identity for Table, Search for Text)
    chunks = splitter.split_segments(segments)
    print(f"Chunks produced: {len(chunks)}")
    # Should be at least 3 (assuming no recursion found in text)
    assert len(chunks) >= 3

def test_semantic_splitter():
    print("\nTesting SemanticChunker...")
    text = "Sentence one. Sentence two. Sentence three about different topic. Sentence four."
    doc = Document(page_content=text, metadata={"source": "test_sem"})
    
    # Mock embeddings to force split
    class MockEmbeddings:
        def embed_documents(self, texts):
            # Return vectors. 
            # 0,1 close. 2,3 close. 0,2 far.
            vecs = []
            for t in texts:
                if "different" in t or "four" in t:
                    vecs.append([0.0, 1.0])
                else:
                    vecs.append([1.0, 0.0])
            return vecs
            
    splitter = SemanticChunker(embeddings=MockEmbeddings(), threshold=0.5)
    
    # 1. Document -> Segments
    segments = splitter.split_documents([doc])
    print(f"Segments produced: {len(segments)}")
    for s in segments:
        print(f" - {s.content}")
        
    # Expect 2 groups: [one, two] and [three, four]
    assert len(segments) == 2
    
    # 2. Segment -> Chunks (Standard TextSplitter behavior)
    chunks = splitter.split_segments(segments)
    print(f"Chunks produced: {len(chunks)}")
    assert len(chunks) == 2

if __name__ == "__main__":
    try:
        test_markdown_splitter()
        test_semantic_splitter()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
