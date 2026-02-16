import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.csv_excel import ExcelLoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 08: Excel Workflow

Features Tested:
1. ExcelLoader: Loading .xlsx files (Sheet processing).
2. Indexer: Indexing structured sheet content.
3. VectorStore: Standard retrieval.

Expected Results:
- Loading:
    - Input: "docs/dummy.xlsx"
    - Output: Segments representing Sheets (converted to Markdown/Text).
    - Sample Data: Segment(content="| ID | Value |\n|--|--|...", metadata={'sheet_name': 'Sheet1'})
- Indexing:
    - Input: Sheet Segments.
    - Output: Indexed content.
- Retrieval:
    - Query: "Specific data row"
    - Expected Result: Sheet segment containing the row.
    - Sample Output: "| 123 | Specific data row |"
"""

def main():
    setup_environment()
    print_section("08. Excel Workflow")

    # 2. Load
    xlsx_path = Path(__file__).parent.parent / "docs" / "dummy.xlsx"
    print(f"Loading {xlsx_path}...")
    
    loader = ExcelLoader(str(xlsx_path))
    segments = loader.load()
    print(f"Loaded {len(segments)} sheets as segments.")

    # 3. Index
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store("chroma", embeddings, "08_excel_basic")
    
    indexer = Indexer(vector_store, embeddings)
    indexer.index(segments)

    # 4. Retrieve
    print("Retrieving from Excel content...")
    results = vector_store.similarity_search("Specific data row", k=1)
    if results:
         print(f"Row Match: {results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()
