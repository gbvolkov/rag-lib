import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.core.domain import SegmentType

@patch("rag_lib.loaders.pdf.camelot")
def test_pdf_table_extraction(mock_camelot):
    """Cycle 3.4: Verify PDFLoader uses Camelot to extract tables as Markdown."""
    
    # Mock Camelot output: List of Table objects, each having a .df attribute
    mock_table = MagicMock()
    mock_table.df = pd.DataFrame({
        "Col1": ["Val1"],
        "Col2": ["Val2"]
    })
    
    # camelot.read_pdf returns a TableList (list-like)
    mock_camelot.read_pdf.return_value = [mock_table]
    
    # Initialize Loader
    loader = PDFLoader("dummy.pdf")
    
    # Load
    segments = loader.load()
    
    # Assertions
    assert len(segments) == 1
    seg = segments[0]
    
    assert seg.type == SegmentType.TABLE
    assert seg.original_format == "markdown"
    
    # Check content is markdown table
    # Tabulate adds alignment spaces, so we check for components
    assert "| Col1" in seg.content
    assert "| Col2" in seg.content
    assert "| Val1" in seg.content
    assert "| Val2" in seg.content
    assert "---" in seg.content # Separator existence
    
    # Check camelot call args
    mock_camelot.read_pdf.assert_called_with(
        "dummy.pdf", 
        pages='all', 
        backend='poppler'
    )
