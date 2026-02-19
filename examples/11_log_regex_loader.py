import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.regex import RegexHierarchyLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 11: Log Regex Loader Workflow

Features Tested:
1. RegexHierarchyLoader: Loading raw log text as Document objects.
2. RegexHierarchySplitter: Splitting logs into hierarchical segments via timestamp regex.
3. Indexer + VectorStore: Indexing and semantic retrieval over log segments.
"""

def main() -> None:
    setup_environment()
    print_section("11. Log Regex Loader Workflow")

    patterns = [(1, r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")]
    docs_dir = Path(__file__).parent.parent / "docs"
    log_path = docs_dir / "anon.log"


    if not log_path:
        print("No log file found for this example. Checked docs/ and examples/ candidates.")
        return

    print_section("1. Loading Logs")
    print(f"Loading {log_path.name} using RegexHierarchyLoader...")
    loader = RegexHierarchyLoader(file_path=str(log_path), patterns=patterns)
    docs = loader.load()

    if not docs:
        print("No documents loaded from log source. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw log length: {len(docs[0].page_content)} characters.")
    print(f"Sample metadata: {docs[0].metadata}")
    save_json_results(docs, "11_log_regex_loader", "loaded_documents")

    print_section("2. Regex Segmentation")
    splitter = RegexHierarchySplitter(patterns=patterns)
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s).")
    if not segments:
        print("No segments produced after regex splitting. Exiting.")
        return

    print(f"Sample segment: {segments[0].content[:180]}...")
    print(f"Sample metadata: {segments[0].metadata}")
    save_json_results(segments, "11_log_regex_loader", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="11_log_regex_loader",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} segment(s) into '11_log_regex_loader'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "Credit card"
    print(f"Query: {query}")
    retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    results = retriever.invoke(query)
    save_json_results(results, "11_log_regex_loader", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"title={metadata.get('title', 'n/a')} "
            f"source={metadata.get('source', 'n/a')}"
        )


if __name__ == "__main__":
    main()
