import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.json import JsonSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.data_loaders import JsonLoader
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 09: JSON Hybrid Workflow

Features Tested:
1. JsonLoader: Loading raw JSON as a document.
2. JsonSplitter: Splitting top-level JSON items into segments.
3. Indexer + Chroma: Indexing JSON segments with metadata.
4. Hybrid Retrieval: Vector search with metadata filter.
"""


def _print_results(label: str, results: list) -> None:
    print(f"{label}: {len(results)}")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"json_index={metadata.get('json_index', 'n/a')} "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"source={metadata.get('source', 'n/a')}"
        )


def main() -> None:
    setup_environment()
    print_section("09. JSON Hybrid Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    json_path = docs_dir / "QA_data.json"
    if not json_path.exists():
        print(f"JSON file not found: {json_path}")
        return

    print_section("1. Loading JSON")
    print(f"Loading {json_path.name} using JsonLoader...")
    loader = JsonLoader(str(json_path), ensure_ascii=False)
    docs = loader.load()

    if not docs:
        print("No documents loaded from JSON. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw JSON length: {len(docs[0].page_content)} characters.")
    save_json_results(docs, "09_json_hybrid", "loaded_documents")

    print_section("2. JSON Segmentation")
    splitter = JsonSplitter(jq_schema=".", ensure_ascii=False)
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s) from JSON.")
    if not segments:
        print("No segments produced after JSON splitting. Exiting.")
        return

    print(f"Sample segment: {segments[0].content[:180]}...")
    print(f"Sample metadata: {segments[0].metadata}")
    save_json_results(segments, "09_json_hybrid", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="09_json_hybrid",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} segment(s) into '09_json_hybrid'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Hybrid Retrieval")
    query = "WEB:CRM"
    metadata_filter = {"json_index": 0}

    print(f"Query: {query}")
    print("Running baseline vector search...")
    baseline_results = vector_store.similarity_search(query, k=3)
    save_json_results(baseline_results, "09_json_hybrid", "retrieved_results_basic")
    _print_results("Baseline Results", baseline_results)

    print(f"Running filtered vector search with filter={metadata_filter}...")
    filtered_results = vector_store.similarity_search(query, k=3, filter=metadata_filter)
    save_json_results(filtered_results, "09_json_hybrid", "retrieved_results_filtered")
    _print_results("Filtered Results", filtered_results)


if __name__ == "__main__":
    main()
