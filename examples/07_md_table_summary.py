import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.core.domain import SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.data_loaders import TextLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 07 (Markdown): Markdown Table Summary Workflow

Features Tested:
1. TextLoader: Loading a markdown document with mixed text and tables.
2. MarkdownTableSplitter: Extracting table/text segments.
3. Splitter-level summaries: table summaries stored in metadata.
4. Optional summary injection into page_content for indexing/search.
5. Indexer + VectorStore: Indexing table segments.
6. Vector Retriever: Querying summarized table content.
"""


def main() -> None:
    setup_environment()
    print_section("07. Markdown Table Summary Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    md_path = docs_dir / "07_md_table_summary_input_ru.md"
    if not md_path.exists():
        print(f"Markdown file not found: {md_path}")
        return

    print_section("1. Loading Markdown")
    print(f"Loading {md_path.name} using TextLoader...")
    loader = TextLoader(str(md_path))
    docs = loader.load()

    if not docs:
        print("No documents loaded from markdown. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Markdown length: {len(docs[0].page_content)} characters.")
    save_json_results(docs, "07_md_table_summary", "loaded_documents")

    print_section("2. Splitting Markdown Tables + Summaries")
    llm = create_llm(provider="openai", model_name="gpt-4.1-nano", temperature=0, streaming=False)
    summarizer = LLMTableSummarizer(llm=llm)
    splitter = MarkdownTableSplitter(
        split_table_rows=True,
        max_rows_per_chunk=1,
        summarizer=summarizer,
        summarize_table=True,
        summarize_chunks=False,
        inject_summaries_into_content=True,
    )

    segments = splitter.split_documents(docs)
    table_segments = [seg for seg in segments if seg.type == SegmentType.TABLE]
    text_segments = [seg for seg in segments if seg.type == SegmentType.TEXT]

    print(f"Total segments: {len(segments)}")
    print(f"Table segments: {len(table_segments)}")
    print(f"Text segments: {len(text_segments)}")
    if not table_segments:
        print("No table segments found. Exiting.")
        return

    sample_meta = table_segments[0].metadata or {}
    print(f"Table summary exists: {'table_summary' in sample_meta}")
    print(f"Injected content prefix: {table_segments[0].content[:180]}...")
    save_json_results(segments, "07_md_table_summary", "segments")

    print_section("3. Indexing Table Segments")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    collection_name = f"07_md_table_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(table_segments)} table segment(s) into '{collection_name}'...")
    indexer.index(table_segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "какой канал имеет лучшую конверсию"
    print(f"Query: {query}")
    retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    results = retriever.invoke(query)
    save_json_results(results, "07_md_table_summary", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"segment_id={doc.metadata.get('segment_id', 'n/a')} "
            f"is_table={doc.metadata.get('is_table', 'n/a')} "
            f"table_summary={'table_summary' in doc.metadata} "
            f"source={doc.metadata.get('source', 'n/a')}"
        )


if __name__ == "__main__":
    main()
