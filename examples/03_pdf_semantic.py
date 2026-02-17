import sys
from pathlib import Path
from dotenv import load_dotenv

# 0. Setup Env EARLY
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results

# 1. Imports - Library Abstractions ONLY
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.pymupdf import PyMuPDFLoader
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from rag_lib.embeddings.factory import get_embeddings_model
from rag_lib.retrieval.retrievers import get_vector_retriever
from rag_lib.retrieval.composition import create_reranking_retriever

"""
E2E Example 03: PDF Semantic Workflow (with Reranking)

Features Tested:
1. PyMuPDFLoader / PDFLoader: Extracting formatted text from PDF (as one Document).
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
    docs = []
    used_loader = "unknown"

    # Prefer PyMuPDF markdown to keep formatting in one Document.
    try:
        pymupdf_loader = PyMuPDFLoader(str(pdf_path), output_format="markdown")
        docs = pymupdf_loader.load()
        used_loader = "PyMuPDFLoader(markdown)"
    except Exception as e:
        print(f"PyMuPDFLoader unavailable/failed: {e}. Falling back to PDFLoader(mode='text').")

    if not docs:
        loader = PDFLoader(str(pdf_path), mode="text")
        docs = loader.load()
        used_loader = "PDFLoader"

    print(f"Loaded via {used_loader}.")
    print(f"Loaded {len(docs)} documents (pages/files).")

    # 3. Chunking Pipeline:
    #    3.1 Structured split: split_documents(...) -> List[Segment]
    #    3.2 Semantic split:   split_segments(...)  -> List[Segment]
    print("Structured + Semantic Chunking (this may take a moment)...")
    embeddings = get_embeddings_model(provider="openai")

    # Stage 1: Structured splitter by heading-like lines.
    # A conservative set that works for markdown headings, numbered headings,
    # and uppercase section titles.
    structured_splitter = RegexHierarchySplitter(
        patterns=[
            (1, r"^\s*#\s+(.+)$"),
            (2, r"^\s*##\s+(.+)$"),
            (3, r"^\s*###\s+(.+)$"),
            (1, r"^\s*\*\*(.+?)\*\*\s*$"),  # Markdown bold-only line as heading
        ],
        exclude_patterns=[
            r"^\s*\d+\s*$",  # Standalone page numbers if still present
        ],
        include_parent_content=False,
    )

    chunker = SemanticChunker(
        embeddings=embeddings,
        threshold_type="fixed",
        threshold=0.8,
        window_size=4,
    )

    # For speed in demo, maybe limit to first few pages if document is huge?
    # But SemanticChunker works best on continuous text. 
    # If using 'statement.pdf' (small), full doc is fine. 
    # If 'russkij-yazyk' is large, we might want to slice `docs`.
    # Let's slice docs to first 10 pages max for demo speed.
    docs_to_process = docs[:10]
    
    structured_segments = structured_splitter.split_documents(docs_to_process)
    print(
        f"Structured splitter created {len(structured_segments)} segments "
        f"from {len(docs_to_process)} input docs."
    )
    save_json_results(structured_segments, "03_pdf_semantic", "structured_segments")

    segments = []
    for structured_seg in structured_segments:
        segments.extend(chunker.split_segments([structured_seg]))

    print(f"Semantic Chunker created {len(segments)} segments from {len(structured_segments)} structured segments.")
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
    base_retriever = get_vector_retriever(vector_store=vector_store, k=10)#, search_type="similarity_score_threshold", score_threshold=0.2)
    
    query = "Что такое морфология?" # Relevant for Russian Grammar PDF
    # Determine appropriate query based on file loaded? 
    if "statement" in str(pdf_path).lower():
         query = "balance summary" 

    print(f"Query: {query}")

    base_results = base_retriever.invoke(query)
    save_json_results(base_results, "03_pdf_semantic", "basic_retrieved_results")


    try:
        # Wrap with Reranker
        # Note: 'device="cpu"' is safer for general environments without CUDA
        reranker = create_reranking_retriever(
            base_retriever,     
            top_n=3,
            reranker_model="BAAI/bge-reranker-v2-m3", # Good multilingual reranker
            max_score_ratio=0.08, #ratio of max similarity score
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

if __name__ == "__main__":
    main()


