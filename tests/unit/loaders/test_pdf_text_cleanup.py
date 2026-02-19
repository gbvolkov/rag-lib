from unittest.mock import MagicMock, patch

from rag_lib.loaders.pdf import PDFLoader


@patch("rag_lib.loaders.pdf.pypdf.PdfReader")
def test_pdf_text_cleanup_removes_repeated_headers_and_page_markers(mock_reader):
    page1 = MagicMock()
    page1.extract_text.return_value = (
        "Common Header Line\n"
        "Shared Footer Marker\n"
        "34\n"
        "Morphology chapter.\n"
    )
    page2 = MagicMock()
    page2.extract_text.return_value = (
        "Common Header Line\n"
        "Shared Footer Marker\n"
        "35\n"
        "Morphological norms section.\n"
    )
    mock_reader.return_value.pages = [page1, page2]

    loader = PDFLoader("dummy.pdf", parse_mode="text")
    docs = loader.load()

    assert len(docs) == 1
    text = docs[0].page_content

    assert "Common Header Line" not in text
    assert "Shared Footer Marker" not in text
    assert "\n34\n" not in f"\n{text}\n"
    assert "\n35\n" not in f"\n{text}\n"
    assert "Morphology chapter." in text
    assert "Morphological norms section." in text


@patch("rag_lib.loaders.pdf.pypdf.PdfReader")
def test_pdf_text_cleanup_keeps_non_repeated_lines(mock_reader):
    page1 = MagicMock()
    page1.extract_text.return_value = "Unique heading A\nPage text A."
    page2 = MagicMock()
    page2.extract_text.return_value = "Unique heading B\nPage text B."
    mock_reader.return_value.pages = [page1, page2]

    loader = PDFLoader("dummy.pdf", parse_mode="text")
    docs = loader.load()

    text = docs[0].page_content
    assert "Unique heading A" in text
    assert "Unique heading B" in text
    assert "Page text A." in text
    assert "Page text B." in text

