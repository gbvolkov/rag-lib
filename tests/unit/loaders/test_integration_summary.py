import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.csv_excel import CSVLoader, ExcelLoader
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.summarizers.table import TableSummarizer

# Mock Summarizer
class MockSummarizer(TableSummarizer):
    def summarize(self, markdown_table: str) -> str:
        return f"Summary of: {markdown_table[:10]}..."

@pytest.fixture
def mock_summarizer():
    return MockSummarizer()

def test_csv_loader_with_summary(tmp_path, mock_summarizer):
    # Create dummy CSV
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df.to_csv(csv_file, index=False)
    
    loader = CSVLoader(str(csv_file), summarizer=mock_summarizer)
    segments = loader.load()
    
    assert len(segments) == 1
    seg = segments[0]
    assert "summary" in seg.metadata
    assert "Summary of:" in seg.metadata["summary"]
    assert "col1" in seg.content

def test_excel_loader_with_summary(tmp_path, mock_summarizer):
    # Create dummy Excel
    # Requires openpyxl
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")
        
    xlsx_file = tmp_path / "test.xlsx"
    df = pd.DataFrame({"col1": [10, 20], "col2": ["x", "y"]})
    df.to_excel(xlsx_file, index=False)
    
    loader = ExcelLoader(str(xlsx_file), summarizer=mock_summarizer)
    segments = loader.load()
    
    assert len(segments) == 1
    seg = segments[0]
    assert "summary" in seg.metadata
    assert "Summary of:" in seg.metadata["summary"]

@patch("rag_lib.loaders.pdf.camelot")
def test_pdf_loader_with_summary(mock_camelot, tmp_path, mock_summarizer):
    # Mock Camelot behavior
    if mock_camelot is None:
        pytest.skip("rag_lib.loaders.pdf.camelot could not be patched (module likely None)")
        
    # Setup mock table
    mock_table = MagicMock()
    mock_table.df = pd.DataFrame({"Header": ["Val1"], "Header2": ["Val2"]})
    mock_table.page = 1
    
    mock_camelot.read_pdf.return_value = [mock_table]
    
    loader = PDFLoader("dummy.pdf", summarizer=mock_summarizer)
    segments = loader.load()
    
    assert len(segments) == 1
    seg = segments[0]
    assert "summary" in seg.metadata
    assert "Summary of:" in seg.metadata["summary"]
    # Check fallback / markdown
    assert "Header" in seg.content

def test_pdf_loader_poppler_error_msg():
    # Test that missing poppler raises clearer error
    with patch("rag_lib.loaders.pdf.camelot") as mock_camelot:
        mock_camelot.read_pdf.side_effect = Exception("Unable to find poppler")
        
        loader = PDFLoader("dummy.pdf")
        with pytest.raises(RuntimeError) as excinfo:
            loader.load()
        
        assert "Ensure 'poppler' is installed" in str(excinfo.value)
