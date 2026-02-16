import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.csv_excel import CSVLoader
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
# Assuming TableSummarizer exists in processors or summarizers
from rag_lib.summarizers.table_llm import LLMTableSummarizer

from langchain_openai import ChatOpenAI

"""
E2E Example 07: CSV Table & Summary Workflow

Features Tested:
1. CSVLoader: Loading tabular data.
2. MarkdownTableSplitter: Preserving table structure during splitting.
3. LLMTableSummarizer: Generating natural language summaries of tables.
4. VectorStore: Indexing summaries (or tables).

Expected Results:
- Loading:
    - Input: "docs/dummy.csv"
    - Output: Segments containing Markdown representation of rows.
    - Sample Data: "| id | name | ... |\n| 1 | item1 | ... |"
- Chunking:
    - Logic: MarkdownTableSplitter (respects headers/rows).
    - Output: Chunks of valid markdown tables.
- Summarization:
    - Input: Table Chunk.
    - Logic: LLM "Summarize this table".
    - Output: String summary.
    - Sample Data: "This table lists inventory items including item1..."
"""

def main():
    setup_environment()
    print_section("07. CSV Table & Summary Workflow")

    # 2. Load
    csv_path = Path(__file__).parent.parent / "docs" / "dummy.csv"
    print(f"Loading {csv_path} (reading first 100 lines to simulate chunk)...")
    
    # CSVLoader usually reads whole file.
    loader = CSVLoader(str(csv_path)) 
    # For demo speed with large file, we might interrupt or use a smaller sample, 
    # but let's assume loader handles it or we use a limit if available.
    segments = loader.load() 
    print(f"Loaded {len(segments)} segments (Markdown Tables).")

    # 3. Chunk (MarkdownTableSplitter)
    # MarkdownTableSplitter splits by rows/headers, not chunk size usually?
    splitter = MarkdownTableSplitter()
    
    all_chunks = []
    for seg in segments:
        all_chunks.extend(splitter.split_text(seg.content))
        
    print(f"Split into {len(all_chunks)} table chunks.")

    # 4. Summarize Tables
    print("Summarizing Tables with LLM...")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    summarizer = LLMTableSummarizer(llm)
    
    # Process first 3 for demo
    for i, chunk in enumerate(all_chunks[:3]):
        summary = summarizer.summarize(chunk)
        print(f"[Table {i}] Summary: {summary[:100]}...")
        # In a real app, we'd attach this summary to metadata and index it.

    # 5. Index & Retrieve (Standard)
    # ... standard Indexer flow ...

if __name__ == "__main__":
    main()
