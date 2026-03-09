from __future__ import annotations

from pathlib import Path

import pytest

from _pptx_test_utils import write_pptx
from rag_lib import PPTXLoader as RootPPTXLoader
from rag_lib.loaders import PPTXLoader as PackagePPTXLoader
from rag_lib.loaders.pptx import PPTXLoader


def _presentation_xml(*rel_ids: str) -> str:
    slide_ids = "\n".join(
        f'    <p:sldId id="{256 + index}" r:id="{rel_id}"/>'
        for index, rel_id in enumerate(rel_ids, start=1)
    )
    return f"""
    <p:presentation
      xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <p:sldIdLst>
{slide_ids}
      </p:sldIdLst>
    </p:presentation>
    """


def _presentation_rels(*targets: str) -> str:
    relationships = "\n".join(
        (
            f'  <Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="{target}"/>'
        )
        for index, target in enumerate(targets, start=1)
    )
    return f"""
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{relationships}
    </Relationships>
    """


def _slide_xml(shapes: str) -> str:
    return f"""
    <p:sld
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
      xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
      xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
      xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld>
        <p:spTree>
          <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
          <p:grpSpPr/>
{shapes}
        </p:spTree>
      </p:cSld>
    </p:sld>
    """


def _notes_xml(shapes: str) -> str:
    return f"""
    <p:notes
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
      xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld>
        <p:spTree>
          <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
          <p:grpSpPr/>
{shapes}
        </p:spTree>
      </p:cSld>
    </p:notes>
    """


def _rels_xml(*relationships: tuple[str, str, str, str | None]) -> str:
    rows = []
    for rel_id, rel_type, target, target_mode in relationships:
        target_mode_attr = f' TargetMode="{target_mode}"' if target_mode else ""
        rows.append(
            f'  <Relationship Id="{rel_id}" Type="{rel_type}" Target="{target}"{target_mode_attr}/>'
        )
    return (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(rows)
        + "\n</Relationships>\n"
    )


def _text_shape(shape_id: int, name: str, text_body: str, placeholder_type: str | None = None) -> str:
    placeholder = f'<p:ph type="{placeholder_type}"/>' if placeholder_type else ""
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="{name}"/>
        <p:cNvSpPr/>
        <p:nvPr>{placeholder}</p:nvPr>
      </p:nvSpPr>
      <p:spPr/>
      <p:txBody>
        <a:bodyPr/>
        <a:lstStyle/>
{text_body}
      </p:txBody>
    </p:sp>
    """


def _title_paragraph(text: str) -> str:
    return f"""
    <a:p>
      <a:r><a:rPr lang="en-US"/><a:t>{text}</a:t></a:r>
    </a:p>
    """


def test_pptx_loader_is_exported_from_public_modules() -> None:
    assert RootPPTXLoader is PPTXLoader
    assert PackagePPTXLoader is PPTXLoader


def test_pptx_loader_renders_text_tables_notes_and_structure(tmp_path: Path) -> None:
    parts = {
        "ppt/presentation.xml": _presentation_xml("rId1", "rId2"),
        "ppt/_rels/presentation.xml.rels": _presentation_rels("slides/slide1.xml", "slides/slide2.xml"),
        "ppt/slides/slide1.xml": _slide_xml(
            _text_shape(2, "Title 1", _title_paragraph("Quarterly Review"), "title")
            + _text_shape(
                3,
                "Body 1",
                """
                <a:p>
                  <a:r><a:rPr lang="en-US"/><a:t>Visit </a:t></a:r>
                  <a:r>
                    <a:rPr lang="en-US"><a:hlinkClick r:id="rIdLink"/></a:rPr>
                    <a:t>portal</a:t>
                  </a:r>
                </a:p>
                <a:p>
                  <a:pPr lvl="0"><a:buChar char="*"/></a:pPr>
                  <a:r><a:rPr lang="en-US"/><a:t>Item one</a:t></a:r>
                </a:p>
                <a:p>
                  <a:pPr lvl="1"><a:buAutoNum type="arabicPeriod" startAt="3"/></a:pPr>
                  <a:r><a:rPr lang="en-US"/><a:t>Nested step</a:t></a:r>
                </a:p>
                """,
            )
            + """
            <p:graphicFrame>
              <p:nvGraphicFramePr>
                <p:cNvPr id="4" name="Table 1"/>
                <p:cNvGraphicFramePr/>
                <p:nvPr/>
              </p:nvGraphicFramePr>
              <p:xfrm/>
              <a:graphic>
                <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">
                  <a:tbl>
                    <a:tr h="0">
                      <a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Name</a:t></a:r></a:p></a:txBody></a:tc>
                      <a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Value</a:t></a:r></a:p></a:txBody></a:tc>
                    </a:tr>
                    <a:tr h="0">
                      <a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>Alpha</a:t></a:r></a:p></a:txBody></a:tc>
                      <a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>42</a:t></a:r></a:p></a:txBody></a:tc>
                    </a:tr>
                  </a:tbl>
                </a:graphicData>
              </a:graphic>
            </p:graphicFrame>
            """
        ),
        "ppt/slides/_rels/slide1.xml.rels": _rels_xml(
            (
                "rIdLink",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                "https://example.com",
                "External",
            ),
            (
                "rIdNote",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide",
                "../notesSlides/notesSlide1.xml",
                None,
            ),
        ),
        "ppt/notesSlides/notesSlide1.xml": _notes_xml(
            _text_shape(
                2,
                "Slide Image Placeholder",
                _title_paragraph("ignored"),
                "sldImg",
            )
            + _text_shape(
                3,
                "Notes Text",
                _title_paragraph("Speaker note for slide one."),
                "body",
            )
        ),
        "ppt/slides/slide2.xml": _slide_xml(
            _text_shape(
                2,
                "Title 2",
                """
                <a:p>
                  <a:r><a:rPr lang="en-US" b="1"/><a:t>Agenda</a:t></a:r>
                  <a:br/>
                  <a:r><a:rPr lang="en-US"/><a:t>First line</a:t></a:r>
                  <a:br/>
                  <a:r><a:rPr lang="en-US"/><a:t>Second line</a:t></a:r>
                </a:p>
                """,
                "title",
            )
            + _text_shape(
                3,
                "Body 2",
                """
                <a:p>
                  <a:r><a:rPr lang="ru-RU"/><a:t>\u041f\u0440\u0438\u0432\u0435\u0442</a:t></a:r>
                </a:p>
                """,
            )
        ),
    }

    pptx_path = write_pptx(tmp_path, parts=parts)
    docs = PPTXLoader(str(pptx_path)).load()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source_type"] == "pptx"
    assert doc.metadata["output_format"] == "markdown"
    assert doc.metadata["slide_count"] == 2
    assert "# Slide 1: Quarterly Review" in doc.page_content
    assert "[portal](https://example.com)" in doc.page_content
    assert "- Item one" in doc.page_content
    assert "  3. Nested step" in doc.page_content
    assert "| Name | Value |" in doc.page_content
    assert "### Speaker Notes" in doc.page_content
    assert "Speaker note for slide one." in doc.page_content
    assert "# Slide 2: Agenda" in doc.page_content
    assert "First line" in doc.page_content
    assert "Second line" in doc.page_content
    assert "\u041f\u0440\u0438\u0432\u0435\u0442" in doc.page_content


def test_pptx_loader_emits_image_placeholder_without_summary(tmp_path: Path) -> None:
    parts = {
        "ppt/presentation.xml": _presentation_xml("rId1"),
        "ppt/_rels/presentation.xml.rels": _presentation_rels("slides/slide1.xml"),
        "ppt/slides/slide1.xml": _slide_xml(
            _text_shape(2, "Title 1", _title_paragraph("Gallery"), "title")
            + """
            <p:pic>
              <p:nvPicPr>
                <p:cNvPr id="3" name="Hero Image" descr="Product preview"/>
                <p:cNvPicPr/>
                <p:nvPr/>
              </p:nvPicPr>
              <p:blipFill><a:blip r:embed="rIdImg"/></p:blipFill>
              <p:spPr/>
            </p:pic>
            """
        ),
        "ppt/slides/_rels/slide1.xml.rels": _rels_xml(
            (
                "rIdImg",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "../media/image1.png",
                None,
            ),
        ),
        "ppt/media/image1.png": b"not-a-real-png",
    }

    pptx_path = write_pptx(tmp_path, parts=parts)
    docs = PPTXLoader(str(pptx_path)).load()

    assert len(docs) == 1
    assert "[image: Hero Image]" in docs[0].page_content
    assert "Summary:" not in docs[0].page_content


def test_pptx_loader_summarizes_chart_and_smartart_with_custom_summarizer(tmp_path: Path) -> None:
    class _Summarizer:
        def __init__(self) -> None:
            self.visuals = []

        def summarize(self, visual):
            self.visuals.append(visual)
            return f"summary for {visual.kind}"

    summarizer = _Summarizer()
    parts = {
        "ppt/presentation.xml": _presentation_xml("rId1"),
        "ppt/_rels/presentation.xml.rels": _presentation_rels("slides/slide1.xml"),
        "ppt/slides/slide1.xml": _slide_xml(
            _text_shape(2, "Title 1", _title_paragraph("Metrics"), "title")
            + """
            <p:graphicFrame>
              <p:nvGraphicFramePr>
                <p:cNvPr id="3" name="Revenue Chart"/>
                <p:cNvGraphicFramePr/>
                <p:nvPr/>
              </p:nvGraphicFramePr>
              <p:xfrm/>
              <a:graphic>
                <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">
                  <c:chart r:id="rIdChart"/>
                </a:graphicData>
              </a:graphic>
            </p:graphicFrame>
            <p:graphicFrame>
              <p:nvGraphicFramePr>
                <p:cNvPr id="4" name="Process Diagram"/>
                <p:cNvGraphicFramePr/>
                <p:nvPr/>
              </p:nvGraphicFramePr>
              <p:xfrm/>
              <a:graphic>
                <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/diagram">
                  <dgm:relIds r:dm="rIdSmart"/>
                </a:graphicData>
              </a:graphic>
            </p:graphicFrame>
            """
        ),
        "ppt/slides/_rels/slide1.xml.rels": _rels_xml(
            (
                "rIdChart",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart",
                "../charts/chart1.xml",
                None,
            ),
            (
                "rIdSmart",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/diagramData",
                "../diagrams/data1.xml",
                None,
            ),
        ),
        "ppt/charts/chart1.xml": """
        <c:chartSpace
          xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <c:chart>
            <c:title><c:tx><c:rich><a:p><a:r><a:t>Revenue</a:t></a:r></a:p></c:rich></c:tx></c:title>
            <c:plotArea>
              <c:barChart>
                <c:ser>
                  <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Current</c:v></c:pt></c:strCache></c:strRef></c:tx>
                  <c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt></c:strCache></c:strRef></c:cat>
                  <c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>10</c:v></c:pt><c:pt idx="1"><c:v>14</c:v></c:pt></c:numCache></c:numRef></c:val>
                </c:ser>
              </c:barChart>
            </c:plotArea>
          </c:chart>
        </c:chartSpace>
        """,
        "ppt/diagrams/data1.xml": """
        <dgm:dataModel
          xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <dgm:ptLst>
            <dgm:pt modelId="1"><dgm:t>Collect data</dgm:t></dgm:pt>
            <dgm:pt modelId="2"><dgm:t>Analyze</dgm:t></dgm:pt>
            <dgm:pt modelId="3"><dgm:t>Report</dgm:t></dgm:pt>
          </dgm:ptLst>
          <dgm:cxnLst>
            <dgm:cxn srcId="1" destId="2"/>
            <dgm:cxn srcId="2" destId="3"/>
          </dgm:cxnLst>
        </dgm:dataModel>
        """,
    }

    pptx_path = write_pptx(tmp_path, parts=parts)
    docs = PPTXLoader(
        str(pptx_path),
        summarize_visuals=True,
        visual_summarizer=summarizer,
    ).load()

    assert len(docs) == 1
    content = docs[0].page_content
    assert "[chart: Revenue]" in content
    assert "| Category | Current |" in content
    assert "Summary: summary for chart" in content
    assert "[smartart: Process Diagram]" in content
    assert "- Collect data" in content
    assert "  - Analyze" in content
    assert "Summary: summary for smartart" in content
    assert [visual.kind for visual in summarizer.visuals] == ["chart", "smartart"]
    assert all(visual.slide_title == "Metrics" for visual in summarizer.visuals)


def test_pptx_loader_raises_when_visual_summary_fails(tmp_path: Path) -> None:
    class _FailingSummarizer:
        def summarize(self, visual):
            raise RuntimeError(f"cannot summarize {visual.kind}")

    parts = {
        "ppt/presentation.xml": _presentation_xml("rId1"),
        "ppt/_rels/presentation.xml.rels": _presentation_rels("slides/slide1.xml"),
        "ppt/slides/slide1.xml": _slide_xml(
            _text_shape(2, "Title 1", _title_paragraph("Visuals"), "title")
            + """
            <p:pic>
              <p:nvPicPr>
                <p:cNvPr id="3" name="Preview"/>
                <p:cNvPicPr/>
                <p:nvPr/>
              </p:nvPicPr>
              <p:blipFill><a:blip r:embed="rIdImg"/></p:blipFill>
              <p:spPr/>
            </p:pic>
            """
        ),
        "ppt/slides/_rels/slide1.xml.rels": _rels_xml(
            (
                "rIdImg",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "../media/image1.png",
                None,
            ),
        ),
        "ppt/media/image1.png": b"image-bytes",
    }

    pptx_path = write_pptx(tmp_path, parts=parts)
    loader = PPTXLoader(
        str(pptx_path),
        summarize_visuals=True,
        visual_summarizer=_FailingSummarizer(),
    )

    with pytest.raises(RuntimeError, match="cannot summarize image"):
        loader.load()
