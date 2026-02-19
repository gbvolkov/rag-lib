import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.csv_excel import ExcelLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 08: Excel Basic Workflow

Features Tested:
1. ExcelLoader: Loading .xlsx files into one Document per sheet.
2. MarkdownTableSplitter: Converting sheet documents to typed table/text segments.
3. Indexer + Chroma: Indexing structured sheet segments.
4. Vector Retriever: Querying indexed sheet data.
"""


def main() -> None:
    setup_environment()
    print_section("08. Excel Basic Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    xlsx_path = docs_dir / "08_result.xlsx"
    if not xlsx_path.exists():
        print(f"Excel file not found: {xlsx_path}")
        return

    print_section("1. Loading Excel")
    print(f"Loading {xlsx_path.name} using ExcelLoader...")
    loader = ExcelLoader(str(xlsx_path))
    docs = loader.load()

    if not docs:
        print("No sheet documents loaded from Excel. Exiting.")
        return

    print(f"Loaded {len(docs)} sheet document(s).")
    print(f"First sheet: {docs[0].metadata.get('sheet_name', 'n/a')}")
    print(f"First sheet row_count: {docs[0].metadata.get('row_count', 'n/a')}")
    save_json_results(docs, "08_excel_md_basic", "loaded_documents")

    print_section("2. Table Segmentation")
    splitter = MarkdownTableSplitter(split_table_rows=True, max_rows_per_chunk=3)
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s) from Excel sheets.")
    if not segments:
        print("No segments produced after table splitting. Exiting.")
        return

    print(f"Sample segment: {segments[0].content[:160]}...")
    print(f"Sample metadata: {segments[0].metadata}")
    save_json_results(segments, "08_excel_md_basic", "segments")

    print_section("3. Indexing")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="08_excel_md_basic",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments)} segment(s) into '08_excel_md_basic'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "Тестирование уязвимостей"
    print(f"Query: {query}")
    retriever = create_vector_retriever(vector_store=vector_store, top_k=3)
    results = retriever.invoke(query)
    save_json_results(results, "08_excel_md_basic", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"sheet_name={metadata.get('sheet_name', 'n/a')} "
            f"row_count={metadata.get('row_count', 'n/a')} "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"is_table={metadata.get('is_table', 'n/a')}"
        )


if __name__ == "__main__":
    main()
