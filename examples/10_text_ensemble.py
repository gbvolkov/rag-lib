import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.data_loaders import TextLoader
from rag_lib.retrieval.composition import create_ensemble_retriever
from rag_lib.retrieval.retrievers import create_bm25_retriever, create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 10: Text Ensemble Workflow

Features Tested:
1. TextLoader: Loading text into Document objects.
2. SentenceSplitter: Converting documents into sentence-aware Segment objects.
3. Indexer + VectorStore: Indexing segments for dense retrieval.
4. BM25 + Vector + Ensemble: Sparse, dense, and combined retrieval.
"""


def _print_results(label: str, results: list) -> None:
    print(f"{label}: {len(results)}")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"source={metadata.get('source', 'n/a')} "
            f"chunk_index={metadata.get('chunk_index', 'n/a')}"
        )


def main() -> None:
    setup_environment()
    print_section("10. Text Ensemble Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    text_path = docs_dir / "terms&defs.txt"
    if not text_path.exists():
        print(f"Text file not found: {text_path}")
        return

    print_section("1. Loading Text")
    print(f"Loading {text_path.name} using TextLoader...")
    loader = TextLoader(str(text_path))
    docs = loader.load()

    if not docs:
        print("No documents loaded from text file. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw text length: {len(docs[0].page_content)} characters.")
    save_json_results(docs, "10_text_ensemble", "loaded_documents")

    print_section("2. Sentence Chunking")
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=30, language="auto")
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s).")
    if not segments:
        print("No segments produced after sentence chunking. Exiting.")
        return

    print(f"Sample segment: {segments[0].content[:180]}...")
    print(f"Sample metadata: {segments[0].metadata}")
    save_json_results(segments, "10_text_ensemble", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="10_text_ensemble",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} segment(s) into '10_text_ensemble'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Ensemble Retrieval (BM25 + Vector)")
    bm25_retriever = create_bm25_retriever(documents=segments, top_k=3)
    vector_retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    ensemble_retriever = create_ensemble_retriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    query = "Cost of risk"
    print(f"Query: {query}")

    bm25_results = bm25_retriever.invoke(query)
    vector_results = vector_retriever.invoke(query)
    ensemble_results = ensemble_retriever.invoke(query)

    save_json_results(bm25_results, "10_text_ensemble", "retrieved_results_bm25")
    save_json_results(vector_results, "10_text_ensemble", "retrieved_results_vector")
    save_json_results(ensemble_results, "10_text_ensemble", "retrieved_results_ensemble")

    _print_results("BM25 Results", bm25_results)
    _print_results("Vector Results", vector_results)
    _print_results("Ensemble Results", ensemble_results)


if __name__ == "__main__":
    main()
