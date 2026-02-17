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
from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.chunkers.language import detect_nltk_language
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.raptor import RaptorProcessor
from rag_lib.llm.factory import get_llm
from rag_lib.embeddings.factory import get_embeddings_model
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from rag_lib.retrieval.retrievers import get_vector_retriever

"""
E2E Example 04: PDF RAPTOR Workflow

Features Tested:
1. PDFLoader: Text extraction.
2. RecursiveSplitter: Initial leaf generation.
3. RaptorProcessor: Hierarchical Clustering & Summarization Tree.
4. VectorStore: Indexing tree segments (leaves + summaries).
5. Hierarchical Retrieval: Querying the augmented tree.

Expected Results:
- Loading & Splitting:
    - Input: "docs/Georgy Volkov ru.pdf"
    - Output: Leaf Segments (~10-20 small chunks).
    - Sample Data: Segment(content="Georgy Volkov, Backend Developer...", segment_id="leaf_0")
- RAPTOR Processing:
    - Logic: Embed -> Cluster (GMM/UMAP) -> Summarize (LLM) -> Repeat.
    - Output: List[Segment] containing Leaves (Level 0) and Summaries (Level 1+).
    - Sample Data: 
        Segment(
            content="Summary of clusters regarding experience...", 
            metadata={'raptor_level': 1, 'raptor_child_ids': ['leaf_0', 'leaf_1']}
        )
- Indexing:
    - Input: Full Tree (Leaves + Summaries).
    - Output: All levels indexed.
- Retrieval:
    - Query: "backend developer experience"
    - Expected Result: Matches from both detailed leaves and high-level summaries.
"""

def main():
    setup_environment()
    print_section("04. PDF RAPTOR Workflow")

    # 2. Load
    # Primary PDF for RAPTOR demo
    pdf_path = Path(__file__).parent.parent / "docs" / "Georgy Volkov ru.pdf"

    # Fallback for environments where CV file is missing
    if not pdf_path.exists():
        print(f"Primary PDF not found: {pdf_path}")
        pdf_path = Path(__file__).parent.parent / "docs" / "statement.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}. Please add a PDF file to docs/.")
        return

    print(f"Loading {pdf_path}...")
    loader = PDFLoader(str(pdf_path), mode="text")
    docs = loader.load()
    used_loader = "PDFLoader(mode='text')"

    if not docs:
        print("No documents loaded from PDF. Exiting.")
        return

    raw_text = docs[0].page_content
    if not raw_text.strip():
        print("PDF text extraction returned empty content. Exiting.")
        return

    print(f"Loaded via {used_loader}.")
    print(f"Loaded {len(docs)} documents (pages/files).")
    print(f"Extracted {len(raw_text)} characters.")

    # 3. Initial split to leaves
    # Detect language first and route sentence tokenization accordingly.
    detected_language = detect_nltk_language(raw_text)
    print(f"Detected text language for tokenization: {detected_language}")
    print("Generating leaf segments with SentenceSplitter...")
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=20, language=detected_language)
    leaf_texts = splitter.split_text(raw_text)

    leaf_segments = [
        Segment(
            content=text,
            segment_id=f"leaf_{i}",
            type=SegmentType.TEXT,
            original_format="pdf",
            path=[pdf_path.name],
            metadata={
                "leaf_index": i,
                "source_file": pdf_path.name,
            },
        )
        for i, text in enumerate(leaf_texts)
        if text.strip()
    ]

    print(f"Generated {len(leaf_segments)} leaf segments.")
    save_json_results(leaf_segments, "04_pdf_raptor", "leaf_segments")

    if not leaf_segments:
        print("No leaf segments generated. Exiting.")
        return

    # 4. RAPTOR Processing
    llm = get_llm(provider="openai", model="gpt-4.1-nano", temperature=0, streaming=False)
    embeddings = get_embeddings_model(provider="openai")

    print("Initializing RAPTOR Processor...")
    processor = RaptorProcessor(
        llm=llm,
        embeddings=embeddings,
        max_levels=3,
        # threshold=0.5 # GMM Threshold - Not supported in RaptorProcessor init currently
    )

    # This might take time (Clustering + Summarizing)
    print("Running RAPTOR (Hierarchical Summarization)...")
    try:
        raptor_tree = processor.process_segments(leaf_segments)
        print(f"RAPTOR built tree with {len(raptor_tree)} total segments (Leaves + Summaries).")
    except ImportError as e:
        print(f"RAPTOR Dependencies missing: {e}")
        return
    except Exception as e:
        print(f"RAPTOR processing failed: {e}")
        return

    summary_segments = [seg for seg in raptor_tree if seg.metadata.get("is_raptor_summary")]
    print(f"RAPTOR summary nodes: {len(summary_segments)}")

    if summary_segments:
        level_counts = {}
        for seg in summary_segments:
            level = seg.metadata.get("raptor_level", 0)
            level_counts[level] = level_counts.get(level, 0) + 1
        print("Summary distribution by RAPTOR level:")
        for level in sorted(level_counts):
            print(f"- Level {level}: {level_counts[level]} segments")

    save_json_results(raptor_tree, "04_pdf_raptor", "raptor_tree")

    # 5. Index
    print("\nIndexing into Chroma (04_pdf_raptor)...")
    vector_store = get_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="04_pdf_raptor",
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    indexer.index(raptor_tree)

    # 6. Retrieve (Hierarchical)
    # We can query specific levels or the whole tree
    print("\nRetrieving from RAPTOR Tree...")
    retriever = get_vector_retriever(
        vector_store=vector_store,
        k=3,
        search_type="similarity_score_threshold",
        score_threshold=0.0,
    )

    query = "backend developer experience"
    if "statement" in str(pdf_path).lower():
        query = "balance summary"

    print(f"Query: {query}")
    results = retriever.invoke(query)
    save_json_results(results, "04_pdf_raptor", "retrieved_results")

    if not results:
        print("No retrieval results.")
        return

    for i, res in enumerate(results, start=1):
        metadata = res.metadata or {}
        is_summary = bool(metadata.get("is_raptor_summary"))
        node_type = "SUMMARY" if is_summary else "LEAF"
        level = metadata.get("raptor_level", 0 if not is_summary else "?")
        print(f"[{i}] [{node_type} L{level}] {res.page_content[:150]}...")

if __name__ == "__main__":
    main()
