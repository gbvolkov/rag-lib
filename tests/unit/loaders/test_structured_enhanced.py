from __future__ import annotations

from pathlib import Path

from rag_lib.loaders.docx import DocXLoader
from _docx_test_utils import write_docx


def test_docx_loader_preserves_links_and_images(tmp_path: Path) -> None:
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rIdHyper" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.com" TargetMode="External"/>
      <Relationship Id="rIdImg" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
    </Relationships>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <w:body>
        <w:p>
          <w:hyperlink r:id="rIdHyper"><w:r><w:t>Site</w:t></w:r></w:hyperlink>
        </w:p>
        <w:p><w:r><w:drawing><a:blip r:embed="rIdImg"/></w:drawing></w:r></w:p>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml, rels_xml=rels_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    markdown = docs[0].page_content
    assert "[Site](https://example.com)" in markdown
    assert "[image: image1.png]" in markdown


def test_docx_loader_table_conversion(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Col A</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Col B</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>1</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>2</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    markdown = docs[0].page_content
    assert "| Col A | Col B |" in markdown
    assert "| 1 | 2 |" in markdown