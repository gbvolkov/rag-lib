import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.csv_table import CSVTableSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.csv_excel import CSVLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 07: CSV Table & Summary Workflow

Features Tested:
1. CSVLoader: Loading CSV as normalized table text.
2. CSVTableSplitter: Row-based chunking with header retention per chunk.
3. Splitter-level summaries: table-level + chunk-level metadata summaries.
4. Optional summary injection into page_content for indexing/search.
5. Indexer + VectorStore: Indexing chunked table segments.
6. Vector Retriever: Querying indexed table content.
"""


def main() -> None:
    setup_environment()
    print_section("07. CSV Table & Summary Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    csv_path = docs_dir / "data.csv"
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    print_section("1. Loading CSV")
    print(f"Loading {csv_path.name} using CSVLoader...")
    loader = CSVLoader(str(csv_path), output_format="csv")
    docs = loader.load()

    if not docs:
        print("No documents loaded from CSV. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"CSV text length: {len(docs[0].page_content)} characters.")
    print(f"Rows detected: {docs[0].metadata.get('row_count', 'n/a')}")
    print(f"Detected delimiter: {docs[0].metadata.get('delimiter', 'n/a')}")

    save_json_results(docs, "07_csv_table_summary", "loaded_documents")

    print_section("2. CSV Table Row Splitting + Summaries")
    llm = create_llm(provider="openai", model_name="gpt-4.1-nano", temperature=0, streaming=False)
    summarizer = LLMTableSummarizer(llm=llm)
    splitter = CSVTableSplitter(
        max_rows_per_chunk=2,
        max_chunk_size=500,
        summarizer=summarizer,
        summarize_table=True,
        summarize_chunks=True,
        inject_summaries_into_content=True,
    )
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} table chunk segment(s).")
    if not segments:
        print("No table chunks generated after splitting.")
        return

    sample_meta = segments[0].metadata or {}
    print(f"Table summary exists: {'table_summary' in sample_meta}")
    print(f"Chunk summary exists: {'chunk_summary' in sample_meta}")
    print(f"Injected content prefix: {segments[0].content[:160]}...")

    save_json_results(segments, "07_csv_table_summary", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    collection_name = f"07_csv_table_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} table segment(s) into '{collection_name}'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "Продукт уровень 3 ТУРИСТИЧЕСКАЯ"
    print(f"Query: {query}")
    retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    results = retriever.invoke(query)
    save_json_results(results, "07_csv_table_summary", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"segment_id={doc.metadata.get('segment_id', 'n/a')} "
            f"is_table={doc.metadata.get('is_table', 'n/a')} "
            f"table_summary={'table_summary' in doc.metadata} "
            f"chunk_summary={'chunk_summary' in doc.metadata} "
            f"source={doc.metadata.get('source', 'n/a')}"
        )


if __name__ == "__main__":
    main()
