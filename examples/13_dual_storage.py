import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results

from langchain_core.stores import InMemoryStore

from rag_lib.chunkers.token import TokenTextSplitter
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 13: Dual Storage Workflow

Features Tested:
1. Parent Segment creation with rag_lib Segment objects.
2. Child chunk generation with TokenTextSplitter (parent/child linkage).
3. Indexer dual-storage indexing: vector_store for children + doc_store for parents.
4. Scored dual retrieval: query child vectors, hydrate parent documents.
"""


def main() -> None:
    setup_environment()
    print_section("13. Dual Storage Workflow")

    print_section("1. Parent Segments")
    parent_segments = [
        Segment(
            segment_id="doc_rag",
            content=(
                "Retrieval Augmented Generation (RAG) combines retrieval and generation. "
                "A retriever finds relevant external context and the generator uses that context "
                "to produce grounded responses with lower hallucination risk."
            ),
            type=SegmentType.TEXT,
            original_format="text",
            path=["dual_storage_demo.txt"],
            metadata={"title": "RAG Definition", "topic": "rag"},
        ),
        Segment(
            segment_id="doc_indexing",
            content=(
                "Dual storage keeps compact searchable chunks in a vector index while storing "
                "full parent documents in a document store. Retrieval finds chunk matches first "
                "and then hydrates the associated parent document."
            ),
            type=SegmentType.TEXT,
            original_format="text",
            path=["dual_storage_demo.txt"],
            metadata={"title": "Dual Storage Indexing", "topic": "indexing"},
        ),
    ]
    print(f"Prepared {len(parent_segments)} parent segment(s).")
    print(f"Sample parent_id: {parent_segments[0].segment_id}")
    save_json_results(parent_segments, "13_dual_storage", "parent_segments")

    print_section("2. Child Chunking")
    splitter = TokenTextSplitter(chunk_size=35, chunk_overlap=8, model_name="cl100k_base")
    segments = splitter.split_segments(parent_segments)

    if not segments:
        print("No child segments generated. Exiting.")
        return

    print(f"Generated {len(segments)} child segment(s).")
    print(f"Sample child parent_id: {segments[0].metadata.get('parent_id')}")
    print(f"Sample child text: {segments[0].content[:120]}...")
    save_json_results(segments, "13_dual_storage", "segments")

    print_section("3. Indexing")
    model_name = "text-embedding-3-small"
    embeddings = create_embeddings_model(provider="openai", model_name=model_name)
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="13_dual_storage",
        cleanup=True,
    )
    doc_store = InMemoryStore()

    indexer = Indexer(vector_store=vector_store, embeddings=embeddings, doc_store=doc_store)
    print(f"Embedding model_name: {model_name}")
    print(
        f"Indexing {len(segments)} child segment(s) and "
        f"{len(parent_segments)} parent segment(s)..."
    )
    indexer.index(segments, parent_segments=parent_segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    top_k = 5
    query = "What is retrieval augmented generation and why use it?"
    print(f"Query: {query}")
    print(f"top_k: {top_k}")

    retriever = create_scored_dual_storage_retriever(
        vector_store=vector_store,
        doc_store=doc_store,
        id_key="parent_id",
        search_kwargs={"k": top_k},
        search_type=SearchType.similarity_score_threshold,
        score_threshold=0.0,
    )
    results = retriever.invoke(query)
    save_json_results(results, "13_dual_storage", "retrieved_results")

    print(f"Retrieved {len(results)} document(s).")
    if not results:
        return

    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] parent_id={metadata.get('parent_id', 'n/a')}")
        print(f"    score={metadata.get('similarity_score', 'n/a')}")
        print(f"    {doc.page_content[:180]}...")


if __name__ == "__main__":
    main()
