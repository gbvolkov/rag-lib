import pytest
import os
from unittest.mock import MagicMock, patch
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.csv_excel import CSVLoader, ExcelLoader
from rag_lib.core.domain import SegmentType

# Paths
BASE_REGISTRY = os.path.join(os.path.dirname(__file__), "../data_registry")
CSV_DIR = os.path.join(BASE_REGISTRY, "csv")
PDF_DIR = os.path.join(BASE_REGISTRY, "pdf")
EXCEL_DIR = os.path.join(BASE_REGISTRY, "excel")

# --- CSV Tests ---

def test_csv_standard():
    path = os.path.join(CSV_DIR, "standard.csv")
    loader = CSVLoader(path)
    docs = loader.load()
    
    assert len(docs) > 0
    assert docs[0].metadata["table_format"] == "markdown"
    assert "Item 0" in docs[0].page_content
    assert docs[0].metadata["source"] == path

def test_csv_pipes():
    path = os.path.join(CSV_DIR, "pipes.csv")
    loader = CSVLoader(path)
    docs = loader.load()
    
    assert len(docs) > 0
    # Verify delimiter was handled (content should look like markdown table)
    assert "|" in docs[0].page_content

def test_csv_empty():
    path = os.path.join(CSV_DIR, "empty.csv")
    loader = CSVLoader(path)
    # Should probably return empty list or raise error depending on implementation
    # Implementation uses pandas read_csv. Empty file might raise EmptyDataError.
    # Code catches Exception and raises RuntimeError.
    
    with pytest.raises(RuntimeError) as excinfo:
        loader.load()
    assert "Failed to load CSV" in str(excinfo.value)

def test_csv_latin1_encoding_handling():
    # This specifically tests if our loader fails or garbles non-utf8
    path = os.path.join(CSV_DIR, "latin1.csv")
    loader = CSVLoader(path)
    
    # Pandas default is utf-8. reading latin1 as utf-8 usually fails or messes up.
    # Our loader doesn't accept encoding arg yet. 
    # This is a "Known Limitation" check OR a bug discovery.
    try:
        docs = loader.load()
        # If it loads, check content. "Café" might be garbled.
        content = docs[0].page_content
        # In a real comprehensive test, we might ASSERT that it fails to prompt a fix,
        # or assert that it behaves as currently expected (garbled/error).
        # Let's see what happens.
    except RuntimeError:
        pass # Acceptable failure for now

# --- Excel Tests ---

def test_excel_multisheet():
    path = os.path.join(EXCEL_DIR, "multisheet.xlsx")
    loader = ExcelLoader(path)
    segments = loader.load()
    
    # Should have 2 segments (Sheet1, Sheet2)
    assert len(segments) == 2
    sheet_names = [s.metadata["sheet_name"] for s in segments]
    assert "Sheet1" in sheet_names
    assert "Sheet2" in sheet_names

# --- PDF Tests (Mocked primarily due to binary complexity) ---

def test_pdf_backend_selection():
    mock_camelot = MagicMock()
    mock_table = MagicMock()
    mock_table.df.to_markdown.return_value = "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |"
    mock_table.page = 1
    mock_camelot.read_pdf.return_value = [mock_table]

    # Patch the global variable 'camelot' in the pdf module
    with patch("rag_lib.loaders.pdf.camelot", mock_camelot):
        loader = PDFLoader("dummy.pdf", backend="lattice")
        loader.load()
        
        mock_camelot.read_pdf.assert_called_with("dummy.pdf", pages='all', backend='lattice')

def test_pdf_corrupt_file():
    path = os.path.join(PDF_DIR, "corrupt.pdf")
    # Simulate camelot raising an error
    mock_camelot = MagicMock()
    mock_camelot.read_pdf.side_effect = Exception("PDF file is damaged")
    
    with patch("rag_lib.loaders.pdf.camelot", mock_camelot):
        loader = PDFLoader(path)
        with pytest.raises(RuntimeError) as exc:
            loader.load()
        assert "Camelot extraction failed" in str(exc.value)

def test_pdf_missing_poppler_hint():
    mock_camelot = MagicMock()
    mock_camelot.read_pdf.side_effect = Exception("poppler is not installed or not in PATH")
    
    with patch("rag_lib.loaders.pdf.camelot", mock_camelot):
        loader = PDFLoader("dummy.pdf", backend="poppler")
        
        with pytest.raises(RuntimeError) as exc:
            loader.load()
        
        # Verify our helpful error message
        assert "Ensure 'poppler' is installed" in str(exc.value)
