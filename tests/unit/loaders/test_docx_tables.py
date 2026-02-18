from __future__ import annotations

import zipfile
from pathlib import Path

from rag_lib.loaders.docx import DocXLoader
from _docx_test_utils import write_docx


def test_extract_table_as_markdown(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:r><w:t>Paragraph before table</w:t></w:r></w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Header 1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Header 2</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Value 1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Value 2</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    markdown = docs[0].page_content
    assert "Paragraph before table" in markdown
    assert "| Header 1 | Header 2 |" in markdown
    assert "| Value 1 | Value 2 |" in markdown


def test_loader_tolerates_invalid_zip(tmp_path: Path) -> None:
    bad_path = tmp_path / "broken.docx"
    bad_path.write_bytes(b"not-a-zip")

    docs = DocXLoader(str(bad_path)).load()
    assert docs == []


def test_loader_tolerates_missing_document_xml(tmp_path: Path) -> None:
    docx_path = tmp_path / "missing_document.xml.docx"
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/styles.xml", "<w:styles xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'/>")

    docs = DocXLoader(str(docx_path)).load()
    assert docs == []