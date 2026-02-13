import pytest
import pandas as pd
from rag_lib.loaders.csv_excel import CSVLoader
from rag_lib.summarizers.table import TableSummarizer

class MockSummarizer(TableSummarizer):
    def summarize(self, markdown_table: str) -> str:
        return "Summary"

def test_auto_detect_semicolon(tmp_path):
    # Create semicolon CSV
    csv_file = tmp_path / "semi.csv"
    with open(csv_file, "w") as f:
        f.write("col1;col2\n1;a\n2;b")
        
    loader = CSVLoader(str(csv_file))
    segments = loader.load()
    
    assert len(segments) == 1
    # Check if parsed correctly (markdown should differ if parsing failed)
    assert "| col1 | col2 |" in segments[0].content or "|   col1 | col2   |" in segments[0].content
    # Note: tabulate format varies, but columns should be recognized

def test_chunking_large_file(tmp_path):
    # Create CSV with 50 rows
    csv_file = tmp_path / "large.csv"
    df = pd.DataFrame({"col": range(50)})
    df.to_csv(csv_file, index=False)
    
    # Load with chunk_size 10
    loader = CSVLoader(str(csv_file), chunk_size=10)
    segments = loader.load()
    
    # Should yield 5 segments
    assert len(segments) == 5
    assert segments[0].metadata["chunk_index"] == 0
    assert segments[4].metadata["chunk_index"] == 4
    assert segments[0].metadata["row_count"] == 10

def test_summarizer_on_chunks(tmp_path):
    csv_file = tmp_path / "chunks.csv"
    df = pd.DataFrame({"col": range(20)})
    df.to_csv(csv_file, index=False)
    
    summarizer = MockSummarizer()
    loader = CSVLoader(str(csv_file), summarizer=summarizer, chunk_size=10)
    segments = loader.load()
    
    assert len(segments) == 2
    assert segments[0].metadata["summary"] == "Summary"
    assert segments[1].metadata["summary"] == "Summary"
