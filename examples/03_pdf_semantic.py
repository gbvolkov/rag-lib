import sys
from pathlib import Path
from dotenv import load_dotenv

# 0. Setup Env EARLY
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results

# 1. Imports - Library Abstractions ONLY
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from rag_lib.embeddings.factory import get_embeddings_model
from rag_lib.retrieval.retrievers import get_vector_retriever
from rag_lib.retrieval.composition import create_reranking_retriever

"""
E2E Example 03: PDF Semantic Workflow (with Reranking)

Features Tested:
1. PDFLoader (pypdf backend): Extracting text from PDF (as Document).
2. SemanticChunker: Splitting based on embedding similarity (cosine distance).
3. Indexer: Ingesting semantic segments into Chroma Vector Store.
4. RerankingRetriever: Cross-Encoder based re-ranking of results.

Expected Results:
- Loading:
    - Input: "docs/statement.pdf" (or "2025_soo_frp_russkij-yazyk_10_11-2.pdf" if available)
    - Output: List[Document]
- Chunking:
    - Logic: SemanticChunker(threshold=0.8).split_documents(docs)
    - Output: List[Segment]
- Indexing:
    - Output: Indexed in Chroma ('03_pdf_semantic').
- Retrieval:
    - Query: "Что такое морфология?"
    - Stage 1: Vector Search (Top-10) -> Broad recall.
    - Stage 2: Cross-Encoder Reranking (Top-3) -> High precision.
"""

def main():
    setup_environment()
    print_section("03. PDF Semantic Workflow (with Reranking)")

    # 2. Load
    # Primary PDF for semantic demo (Russian Grammar textbook)
    pdf_path = Path(__file__).parent.parent / "docs" / "2025_soo_frp_russkij-yazyk_10_11-2.pdf"
    
    # Fallback/Alternative for testing if big file missing
    if not pdf_path.exists():
         print(f"Primary PDF not found: {pdf_path}")
         pdf_path = Path(__file__).parent.parent / "docs" / "statement.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}. Please add a PDF file to docs/.")
        return

    print(f"Loading {pdf_path}...")
    loader = PDFLoader(str(pdf_path), mode="text")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents (pages/files).")

    # 3. Chunk (Semantic)
    print("Semantic Chunking (this may take a moment)...")
    embeddings = get_embeddings_model(provider="openai")
    
    # SemanticChunker calculates similarity between sentences to find breakpoints
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.8)
    
    # For speed in demo, maybe limit to first few pages if document is huge?
    # But SemanticChunker works best on continuous text. 
    # If using 'statement.pdf' (small), full doc is fine. 
    # If 'russkij-yazyk' is large, we might want to slice `docs`.
    # Let's slice docs to first 10 pages max for demo speed.
    docs_to_process = docs[:10]
    
    segments = chunker.split_documents(docs_to_process)
    print(f"Semantic Chunker created {len(segments)} segments from {len(docs_to_process)} input docs.")
    save_json_results(segments, "03_pdf_semantic", "segments")

    # 4. Index
    print("\nIndexing into Chroma (03_pdf_semantic)...")
    vector_store = get_vector_store(
        provider="chroma", 
        embeddings=embeddings, 
        collection_name="03_pdf_semantic"
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    indexer.index(segments)

    # 5. Retrieve (Reranking)
    print("\nRetrieving with Cross-Encoder Reranking...")
    
    # Base Retriever (Vector)
    base_retriever = get_vector_retriever(vector_store=vector_store, k=10)
    
    query = "Что такое морфология?" # Relevant for Russian Grammar PDF
    # Determine appropriate query based on file loaded? 
    if "statement" in str(pdf_path).lower():
         query = "balance summary" 

    print(f"Query: {query}")

    try:
        # Wrap with Reranker
        # Note: 'device="cpu"' is safer for general environments without CUDA
        reranker = create_reranking_retriever(
            base_retriever, 
            top_n=3,
            reranker_model="BAAI/bge-reranker-base", # Good multilingual reranker
            device="cpu" 
        )
        
        results = reranker.invoke(query)
        
        for i, res in enumerate(results):
            print(f"[{i+1}] {res.page_content[:100]}...")
            
        save_json_results(results, "03_pdf_semantic", "retrieved_results")

    except ImportError:
        print("sentence-transformers not installed. Skipping Reranker.")
        # Fallback to base
        results = base_retriever.invoke(query)
        print("Fallback Vector Results:")
        for r in results[:3]:
            print(f"- {r.page_content[:100]}...")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
