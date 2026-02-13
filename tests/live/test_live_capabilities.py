import os
import pytest
from dotenv import load_dotenv
from rag_lib.embeddings.factory import get_embeddings_model
from rag_lib.llm.factory import get_llm
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.core.domain import Segment

# Load env vars from .env file
load_dotenv()

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not found in environment")
def test_live_semantic_chunking():
    """
    Verifies Semantic Chunking using REAL OpenAI Embeddings.
    """
    print("\n--- Starting Live Semantic Chunking Test ---")
    
    # 1. Initialize Real Embeddings
    print("Initializing OpenAI Embeddings...")
    embeddings = get_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    
    # 2. Initialize Chunker
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.6) # Slightly lower threshold for real world? 
    # Actually 0.6 is default in config, let's stick to it or explicit arg.
    
    # 3. Test Text: Distinct topics (Coding vs Cooking)
    text = (
        "Python is a high-level programming language. It is known for its readability. "
        "Functions are first-class citizens in Python. "
        "The sun was shining brightly on the orchard. The apple trees were full of fruit. "
        "Harvest season is the best time of year for farmers."
    )
    
    # DEBUG: Inject a print in chunker or just subclass/monkeypatch?
    # Let's simple check the embeddings correlation first manually here
    sentences = ["Python is a high-level programming language.", "It is known for its readability."]
    vecs = embeddings.embed_documents(sentences)
    import numpy as np
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim = cos_sim(vecs[0], vecs[1])
    print(f"DEBUG: Similarity between sent 1 and 2: {sim}")
    
    print(f"Chunking text ({len(text)} chars)...")
    segments = chunker.chunk(text)
    
    print(f"Generated {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"Segment {i+1}: {seg.content}")
        
    # Validation
    assert len(segments) >= 2
    
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not found in environment")
def test_live_table_summarization():
    """
    Verifies Table Summarization using REAL OpenAI Chat Model.
    """
    print("\n--- Starting Live Table Summarization Test ---")
    
    try:
        # 1. Initialize Real LLM
        print("Initializing OpenAI Chat Model (gpt-4o-mini)...")
        llm = get_llm(provider="openai", model="base")
        
        # 2. Initialize Summarizer
        summarizer = LLMTableSummarizer(llm=llm)
        
        # 3. Test Table
        table_md = """
| Product | Q1 Sales | Q2 Sales | Growth |
| :--- | :--- | :--- | :--- |
| Widget A | 1000 | 1500 | 50% |
| Widget B | 2000 | 1800 | -10% |
| Widget C | 500  | 5000 | 900% |
"""
        print("Summarizing table...")
        summary = summarizer.summarize(table_md)
        
        print(f"Generated Summary: {summary}")
        
        # Validation
        assert isinstance(summary, str)
        assert len(summary) > 10
        assert "Widget C" in summary or "growth" in summary.lower()
        print("--- Live Table Summarization Test PASSED ---\n")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR in Table Summarization: {e}")
        import traceback
        with open("live_error.log", "w") as f:
            f.write(str(e))
            f.write("\n")
            traceback.print_exc(file=f)
        raise e
