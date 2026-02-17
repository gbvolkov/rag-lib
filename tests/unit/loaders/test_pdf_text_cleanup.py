from unittest.mock import MagicMock, patch

from rag_lib.loaders.pdf import PDFLoader


@patch("rag_lib.loaders.pdf.pypdf.PdfReader")
def test_pdf_text_cleanup_removes_repeated_headers_and_page_markers(mock_reader):
    page1 = MagicMock()
    page1.extract_text.return_value = (
        "Федеральная рабочая программа | Русский язык.\n"
        "10-11 классы (базовый уровень)\n"
        "34\n"
        "Морфология.\n"
    )
    page2 = MagicMock()
    page2.extract_text.return_value = (
        "Федеральная рабочая программа | Русский язык.\n"
        "10-11 классы (базовый уровень)\n"
        "35\n"
        "Морфологические нормы.\n"
    )
    mock_reader.return_value.pages = [page1, page2]

    loader = PDFLoader("dummy.pdf", mode="text")
    docs = loader.load()

    assert len(docs) == 1
    text = docs[0].page_content

    assert "Федеральная рабочая программа | Русский язык." not in text
    assert "10-11 классы (базовый уровень)" not in text
    assert "\n34\n" not in f"\n{text}\n"
    assert "\n35\n" not in f"\n{text}\n"
    assert "Морфология." in text
    assert "Морфологические нормы." in text


@patch("rag_lib.loaders.pdf.pypdf.PdfReader")
def test_pdf_text_cleanup_keeps_non_repeated_lines(mock_reader):
    page1 = MagicMock()
    page1.extract_text.return_value = "Уникальный заголовок A\nТекст страницы A."
    page2 = MagicMock()
    page2.extract_text.return_value = "Уникальный заголовок B\nТекст страницы B."
    mock_reader.return_value.pages = [page1, page2]

    loader = PDFLoader("dummy.pdf", mode="text")
    docs = loader.load()

    text = docs[0].page_content
    assert "Уникальный заголовок A" in text
    assert "Уникальный заголовок B" in text
    assert "Текст страницы A." in text
    assert "Текст страницы B." in text
