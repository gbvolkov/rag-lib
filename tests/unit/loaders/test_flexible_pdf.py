import pytest
from unittest.mock import MagicMock, patch
import os
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.config import Settings

@pytest.fixture
def mock_camelot():
    with patch("rag_lib.loaders.pdf.camelot") as mock:
        yield mock

def test_pdf_backend_default(mock_camelot):
    # Default should be poppler
    loader = PDFLoader("test.pdf")
    assert loader.backend == "poppler"
    
    loader.load()
    mock_camelot.read_pdf.assert_called_with("test.pdf", pages='all', backend='poppler')

def test_pdf_backend_override_init(mock_camelot):
    # Override via init
    loader = PDFLoader("test.pdf", backend="lattice")
    assert loader.backend == "lattice"
    
    loader.load()
    mock_camelot.read_pdf.assert_called_with("test.pdf", pages='all', backend='lattice')

def test_pdf_backend_override_env(mock_camelot):
    # Override via env
    old_env = os.environ.get("INGEST_DEFAULT_PDF_BACKEND")
    os.environ["INGEST_DEFAULT_PDF_BACKEND"] = "lattice"
    try:
        loader = PDFLoader("test.pdf")
        assert loader.backend == "lattice"
        
        loader.load()
        mock_camelot.read_pdf.assert_called_with("test.pdf", pages='all', backend='lattice')
    finally:
        if old_env:
            os.environ["INGEST_DEFAULT_PDF_BACKEND"] = old_env
        else:
            del os.environ["INGEST_DEFAULT_PDF_BACKEND"]
