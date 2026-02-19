import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.qa import QASplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.data_loaders import TextLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 12: QA Loader Workflow

Features Tested:
1. TextLoader: Loading QA text into Document objects.
2. QASplitter: Splitting QA text into Segment objects and extracting question metadata.
3. Indexer + Chroma: Indexing QA segments.
4. Vector Retriever: Querying indexed QA entries.
"""


def main() -> None:
    setup_environment()
    print_section("12. QA Loader Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    qa_path = docs_dir / "interview.txt"
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        return

    print_section("1. Loading QA")
    print(f"Loading {qa_path.name} using TextLoader...")
    loader = TextLoader(str(qa_path))
    docs = loader.load()

    if not docs:
        print("No documents loaded from QA source. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw QA text length: {len(docs[0].page_content)} characters.")
    print(f"Sample metadata: {docs[0].metadata}")
    save_json_results(docs, "12_qa_loader", "loaded_documents")

    print_section("2. QA Segmentation")
    splitter = QASplitter()
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s).")
    if not segments:
        print("No segments produced after QA splitting. Exiting.")
        return

    print(f"Sample segment: {segments[0].content[:180]}...")
    print(f"Sample metadata: {segments[0].metadata}")
    save_json_results(segments, "12_qa_loader", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="12_qa_loader",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} segment(s) into '12_qa_loader'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "graph database experience"
    print(f"Query: {query}")
    retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    results = retriever.invoke(query)
    save_json_results(results, "12_qa_loader", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"question={metadata.get('question', 'n/a')} "
            f"type={metadata.get('type', 'n/a')} "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"source={metadata.get('source', 'n/a')}"
        )


if __name__ == "__main__":
    main()
