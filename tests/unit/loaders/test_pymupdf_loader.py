from pathlib import Path
from unittest.mock import patch

import pytest

from rag_lib.loaders.pymupdf import PyMuPDFLoader


def test_pymupdf_loader_markdown_returns_one_document(tmp_path: Path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("rag_lib.loaders.pymupdf.pymupdf4llm") as mock_llm, patch.object(
        PyMuPDFLoader, "_get_page_count", return_value=3
    ):
        mock_llm.to_markdown.return_value = "# Title\n\nBody"
        docs = PyMuPDFLoader(str(pdf_path), output_format="markdown").load()

    assert len(docs) == 1
    assert docs[0].page_content.startswith("# Title")
    assert docs[0].metadata["output_format"] == "markdown"
    assert docs[0].metadata["page_count"] == 3


def test_pymupdf_loader_html_joins_pages(tmp_path: Path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _Page:
        def __init__(self, html: str):
            self._html = html

        def get_text(self, mode: str) -> str:
            assert mode == "html"
            return self._html

    class _Doc:
        def __init__(self):
            self.pages = [_Page("<p>p1</p>"), _Page("<p>p2</p>")]

        def __iter__(self):
            return iter(self.pages)

        def close(self):
            return None

        def __len__(self):
            return len(self.pages)

    with patch("rag_lib.loaders.pymupdf.fitz") as mock_fitz:
        mock_fitz.open.return_value = _Doc()
        docs = PyMuPDFLoader(str(pdf_path), output_format="html").load()

    assert len(docs) == 1
    assert "<p>p1</p>" in docs[0].page_content
    assert "<p>p2</p>" in docs[0].page_content
    assert docs[0].metadata["output_format"] == "html"
    assert docs[0].metadata["page_count"] == 2


def test_pymupdf_loader_markdown_requires_pymupdf4llm(tmp_path: Path):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("rag_lib.loaders.pymupdf.pymupdf4llm", None):
        loader = PyMuPDFLoader(str(pdf_path), output_format="markdown")
        with pytest.raises(ImportError):
            loader.load()


def test_pymupdf_loader_rejects_unknown_output_format():
    with pytest.raises(ValueError):
        PyMuPDFLoader("dummy.pdf", output_format="xml")
