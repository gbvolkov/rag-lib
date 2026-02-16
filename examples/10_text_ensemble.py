import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.retrieval.composition import create_ensemble_retriever
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 10: Text Ensemble Workflow

Features Tested:
1. SentenceSplitter: Linguistic splitting.
2. BM25Retriever: Sparse keyword search.
3. VectorRetriever: Dense semantic search.
4. EnsembleRetriever: Weighted combination (RRF).

Expected Results:
- Chunking:
    - Logic: Split by sentences.
    - Output: Grammatically complete sentences/paragraphs.
    - Sample Data: "RAG is..."
- Retrieval:
    - Query: "RAG definition"
    - BM25: Finds exact keyword matches ("RAG", "definition").
    - Vector: Finds semantic matches ("Retrieval Augmented...").
    - Ensemble: Reranks both.
    - Sample Output: Top result has high consensus from both retrievers.
"""

# Need standard BM25
from langchain_community.retrievers import BM25Retriever
from rag_lib.core.domain import Segment 

def main():
    setup_environment()
    print_section("10. Text Ensemble Workflow")

    # 2. Load & Split (Sentence)
    txt_path = Path(__file__).parent.parent / "docs" / "terms&defs.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Splitting into sentences...")
    splitter = SentenceSplitter(chunk_size=300)
    chunks = splitter.split_text(text)
    
    # 3. Setup Retrievers
    print("Setting up Ensemble (BM25 + Vector)...")
    
    # BM25 (Sparse)
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 3
    
    # Vector (Dense)
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store("chroma", embeddings, "10_text_ensemble")
    
    # Index for vector
    vector_store.add_texts(chunks)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 4. Ensemble
    ensemble = create_ensemble_retriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] # Equal weight
    )
    
    query = "RAG definition"
    results = ensemble.invoke(query)
    
    print(f"Ensemble Results for '{query}':")
    for r in results:
        print(f"- {r.page_content[:100]}...")

if __name__ == "__main__":
    main()
