from __future__ import annotations

import re
from pathlib import Path

import pytest

from _docx_test_utils import write_docx
from rag_lib.loaders.docx import DocXLoader


def _load_markdown(
    tmp_path: Path,
    *,
    document_xml: str,
    styles_xml: str | None = None,
    numbering_xml: str | None = None,
    rels_xml: str | None = None,
) -> tuple[str, dict]:
    docx_path = write_docx(
        tmp_path,
        document_xml=document_xml,
        styles_xml=styles_xml,
        numbering_xml=numbering_xml,
        rels_xml=rels_xml,
    )
    docs = DocXLoader(str(docx_path)).load()
    assert len(docs) == 1
    return docs[0].page_content, docs[0].metadata


def test_heading_detection_by_style_and_outline(tmp_path: Path) -> None:
    styles_xml = """
    <w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:style w:type="paragraph" w:styleId="BaseHeading">
        <w:name w:val="Base Heading"/>
        <w:pPr><w:outlineLvl w:val="1"/></w:pPr>
      </w:style>
      <w:style w:type="paragraph" w:styleId="ChildHeading">
        <w:name w:val="Child Heading"/>
        <w:basedOn w:val="BaseHeading"/>
      </w:style>
    </w:styles>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p>
          <w:pPr><w:pStyle w:val="ChildHeading"/></w:pPr>
          <w:r><w:t>Styled Heading</w:t></w:r>
        </w:p>
        <w:p>
          <w:pPr><w:outlineLvl w:val="2"/></w:pPr>
          <w:r><w:t>Outlined Heading</w:t></w:r>
        </w:p>
      </w:body>
    </w:document>
    """

    markdown, metadata = _load_markdown(
        tmp_path,
        document_xml=document_xml,
        styles_xml=styles_xml,
    )

    assert "## Styled Heading" in markdown
    assert "### Outlined Heading" in markdown
    assert metadata["source_type"] == "docx"
    assert metadata["output_format"] == "markdown"


def test_heuristic_heading_promotion_for_bold_numbered_lines(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p>
          <w:r>
            <w:rPr><w:b/></w:rPr>
            <w:t>1.2 Scope</w:t>
          </w:r>
        </w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(tmp_path, document_xml=document_xml)
    assert re.search(r"^###\s+1\.2 Scope$", markdown, flags=re.MULTILINE)


def test_inline_formatting_preserved(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p>
          <w:r><w:rPr><w:b/></w:rPr><w:t>Bold</w:t></w:r>
          <w:r><w:t xml:space="preserve"> </w:t></w:r>
          <w:r><w:rPr><w:i/></w:rPr><w:t>Italic</w:t></w:r>
          <w:r><w:t xml:space="preserve"> </w:t></w:r>
          <w:r><w:rPr><w:b/><w:i/></w:rPr><w:t>Both</w:t></w:r>
          <w:r><w:t xml:space="preserve"> </w:t></w:r>
          <w:r><w:rPr><w:strike/></w:rPr><w:t>Strike</w:t></w:r>
          <w:r><w:t xml:space="preserve"> </w:t></w:r>
          <w:r><w:rPr><w:u w:val="single"/></w:rPr><w:t>Under</w:t></w:r>
        </w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(tmp_path, document_xml=document_xml)
    assert "**Bold**" in markdown
    assert "*Italic*" in markdown
    assert "***Both***" in markdown
    assert "~~Strike~~" in markdown
    assert "<u>Under</u>" in markdown

def test_list_rendering_for_bullet_and_ordered_with_nesting(tmp_path: Path) -> None:
    numbering_xml = """
    <w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:abstractNum w:abstractNumId="1">
        <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/></w:lvl>
        <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="o"/></w:lvl>
      </w:abstractNum>
      <w:abstractNum w:abstractNumId="2">
        <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>
      </w:abstractNum>
      <w:num w:numId="10"><w:abstractNumId w:val="1"/></w:num>
      <w:num w:numId="20"><w:abstractNumId w:val="2"/></w:num>
    </w:numbering>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="10"/></w:numPr></w:pPr><w:r><w:t>Bullet A</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="10"/></w:numPr></w:pPr><w:r><w:t>Bullet B</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="20"/></w:numPr></w:pPr><w:r><w:t>First</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="20"/></w:numPr></w:pPr><w:r><w:t>Second</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(
        tmp_path,
        document_xml=document_xml,
        numbering_xml=numbering_xml,
    )

    assert "- Bullet A" in markdown
    assert "  - Bullet B" in markdown
    assert re.search(r"^1\.\s+First$", markdown, flags=re.MULTILINE)
    assert re.search(r"^2\.\s+Second$", markdown, flags=re.MULTILINE)


def test_hyperlink_conversion_external_and_internal(tmp_path: Path) -> None:
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship
        Id="rIdHyper"
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
        Target="https://openai.com/docs"
        TargetMode="External"/>
    </Relationships>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p>
          <w:hyperlink r:id="rIdHyper"><w:r><w:t>OpenAI</w:t></w:r></w:hyperlink>
        </w:p>
        <w:p>
          <w:hyperlink w:anchor="Section1"><w:r><w:t>Jump</w:t></w:r></w:hyperlink>
        </w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(
        tmp_path,
        document_xml=document_xml,
        rels_xml=rels_xml,
    )

    assert "[OpenAI](https://openai.com/docs)" in markdown
    assert "[Jump](#Section1)" in markdown


def test_table_conversion_with_escaping_and_links(tmp_path: Path) -> None:
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship
        Id="rIdCellLink"
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
        Target="https://example.com"
        TargetMode="External"/>
    </Relationships>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc>
              <w:p>
                <w:r><w:rPr><w:b/></w:rPr><w:t>A|B</w:t></w:r>
              </w:p>
            </w:tc>
            <w:tc>
              <w:p>
                <w:hyperlink r:id="rIdCellLink"><w:r><w:t>Link</w:t></w:r></w:hyperlink>
              </w:p>
            </w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(
        tmp_path,
        document_xml=document_xml,
        rels_xml=rels_xml,
    )

    assert "| Name | Value |" in markdown
    assert "| --- | --- |" in markdown
    assert "A\\\\|B" in markdown
    assert "[Link](https://example.com)" in markdown

def test_image_placeholder_emission(tmp_path: Path) -> None:
    rels_xml = """
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship
        Id="rIdImg"
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
        Target="media/image1.png"/>
    </Relationships>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
      <w:body>
        <w:p>
          <w:r>
            <w:drawing><a:blip r:embed="rIdImg"/></w:drawing>
          </w:r>
        </w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(
        tmp_path,
        document_xml=document_xml,
        rels_xml=rels_xml,
    )

    assert "[image: image1.png]" in markdown


def test_utf8_cyrillic_preservation(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:r><w:t>задача и решение</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    markdown, _ = _load_markdown(tmp_path, document_xml=document_xml)
    assert "задача и решение" in markdown
    assert "\\u0437" not in markdown


def _local_docx_candidates() -> list[Path]:
    docs_dir = Path(__file__).resolve().parents[3] / "docs"
    return sorted(docs_dir.glob("*.docx"))[:2]


@pytest.mark.parametrize("docx_path", _local_docx_candidates())
def test_local_docx_golden_markers(docx_path: Path) -> None:
    docs = DocXLoader(str(docx_path)).load()
    assert len(docs) == 1

    markdown = docs[0].page_content
    assert len(markdown) > 200
    assert re.search(r"[А-Яа-яЁё]", markdown)
    assert any(marker in markdown for marker in ["#", "- ", "|", "[image:"])
    if "[image:" in markdown:
        assert "Подготовка Банковской гарантии" in markdown
    else:
        assert "Параметризованные задачи" in markdown