import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from langchain_core.stores import InMemoryStore

from rag_lib.chunkers.html import HTMLSplitter
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.html import HTMLLoader
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 16B: HTML -> Markdown Segments + Dual Retrieval

Features Tested:
1. HTMLLoader in strict html output mode (keeps structured input for HTMLSplitter).
2. HTMLSplitter in markdown mode with heading/list/table-aware conversion.
3. Parent-child dual storage indexing (vector children + doc_store parents).
4. Scored dual retrieval with parent hydration.
"""


def main() -> None:
    setup_environment()
    print_section("16B. HTML -> Markdown Dual Retrieval Workflow")

    html_path = Path(__file__).parent.parent / "docs" / "15_test.html"
    if not html_path.exists():
        print(f"HTML file not found: {html_path}")
        return

    print_section("1. Loading")
    print(f"Loading {html_path.name} using HTMLLoader(output_format='html')...")
    loader = HTMLLoader(str(html_path), output_format="html")
    docs = loader.load()

    if len(docs) != 1:
        print(f"Unexpected document count: {len(docs)}")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw content length: {len(docs[0].page_content)} characters.")
    print(f"Metadata: {docs[0].metadata}")
    save_json_results(docs, "16_html_md", "loaded_documents")

    print_section("2. Parent Segmentation (Markdown mode)")
    logical_splitter = HTMLSplitter(
        output_format="markdown",
        split_table_rows=True,
        max_rows_per_chunk=6,
        use_first_row_as_header=True,
        include_parent_content=False,
    )
    parent_segments = logical_splitter.split_documents(docs)

    if not parent_segments:
        print("No parent segments produced. Exiting.")
        return

    table_count = sum(1 for segment in parent_segments if segment.metadata.get("is_table"))
    print(f"Generated {len(parent_segments)} parent segment(s).")
    print(f"Table parent segments: {table_count}")
    print(f"Sample parent metadata: {parent_segments[0].metadata}")
    print(f"Sample parent content: {parent_segments[0].content[:180]}...")
    save_json_results(parent_segments, "16_html_md", "parent_segments")

    print_section("3. Child Chunking")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    child_segments = child_splitter.split_segments(parent_segments)

    if not child_segments:
        print("No child chunks produced. Exiting.")
        return

    print(f"Generated {len(child_segments)} child segment(s).")
    print(f"Sample child parent_id: {child_segments[0].metadata.get('parent_id')}")
    print(f"Sample child content: {child_segments[0].content[:180]}...")
    save_json_results(child_segments, "16_html_md", "child_segments")

    print_section("4. Indexing (Dual Storage)")
    model_name = "text-embedding-3-small"
    embeddings = create_embeddings_model(provider="openai", model_name=model_name)
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="16_html_md",
        cleanup=True,
    )
    doc_store = InMemoryStore()

    indexer = Indexer(vector_store=vector_store, embeddings=embeddings, doc_store=doc_store)
    print(f"Embedding model_name: {model_name}")
    print(
        f"Indexing {len(child_segments)} child segment(s) and "
        f"{len(parent_segments)} parent segment(s)..."
    )
    indexer.index(child_segments, parent_segments=parent_segments)
    print("Indexing complete.")

    print_section("5. Retrieval (Scored Dual)")
    query = "Which project belongs to A. Novak and what stage is it in?"
    top_k = 6
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
    save_json_results(results, "16_html_md", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] score={metadata.get('similarity_score', 'n/a')}")
        print(f"    parent_id={metadata.get('parent_id', 'n/a')}")
        print(f"    title={metadata.get('title', 'n/a')}")
        print(f"    {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
