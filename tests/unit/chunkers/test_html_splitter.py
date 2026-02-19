from __future__ import annotations

import pytest
from lxml import etree

from rag_lib.chunkers.html import HTMLSplitter
from rag_lib.core.domain import SegmentType


class _MockSummarizer:
    def summarize(self, markdown_table: str) -> str:
        return f"summary(len={len(markdown_table)})"


class _FailingSummarizer:
    def summarize(self, markdown_table: str) -> str:
        raise RuntimeError("summary failed")


def test_html_splitter_heading_hierarchy_metadata() -> None:
    html_text = """
    <html><body>
      <h1>Main</h1>
      <p>Alpha</p>
      <h2>Sub</h2>
      <p>Beta</p>
    </body></html>
    """
    splitter = HTMLSplitter(output_format="markdown")

    segments = splitter.create_segments(html_text)

    assert len(segments) == 2
    assert segments[0].type == SegmentType.TEXT
    assert segments[0].level == 1
    assert segments[0].path == []
    assert segments[0].metadata["title"] == "Main"
    assert segments[0].parent_id is None
    assert "# Main" in segments[0].content

    assert segments[1].type == SegmentType.TEXT
    assert segments[1].level == 2
    assert segments[1].path == ["Main"]
    assert segments[1].metadata["title"] == "Sub"
    assert segments[1].parent_id == segments[0].segment_id
    assert "Beta" in segments[1].content


def test_html_splitter_separates_tables_from_surrounding_text() -> None:
    html_text = """
    <html><body>
      <h1>Main</h1>
      <p>Intro text.</p>
      <ul><li>First</li><li>Second</li></ul>
      <table>
        <tr><th>ID</th><th>Value</th></tr>
        <tr><td>1</td><td>A</td></tr>
      </table>
      <p>After table.</p>
    </body></html>
    """
    splitter = HTMLSplitter(output_format="markdown")

    segments = splitter.create_segments(html_text)

    assert len(segments) == 3
    assert segments[0].type == SegmentType.TEXT
    assert "- First" in segments[0].content
    assert segments[1].type == SegmentType.TABLE
    assert "| ID | Value |" in segments[1].content
    assert segments[2].type == SegmentType.TEXT
    assert "After table." in segments[2].content


def test_html_splitter_table_row_chunk_parity_metadata() -> None:
    html_text = """
    <html><body>
      <table>
        <tr><th>ID</th><th>Value</th></tr>
        <tr><td>1</td><td>A</td></tr>
        <tr><td>2</td><td>B</td></tr>
        <tr><td>3</td><td>C</td></tr>
      </table>
    </body></html>
    """
    splitter = HTMLSplitter(
        output_format="markdown",
        split_table_rows=True,
        max_rows_per_chunk=2,
        summarizer=_MockSummarizer(),
        summarize_table=True,
        summarize_chunks=True,
    )

    segments = splitter.create_segments(html_text)
    table_segments = [segment for segment in segments if segment.type == SegmentType.TABLE]

    assert len(table_segments) == 2
    assert table_segments[0].metadata["table_chunk_total"] == 2
    assert table_segments[0].metadata["data_row_start"] == 0
    assert table_segments[0].metadata["data_row_end"] == 2
    assert table_segments[1].metadata["data_row_start"] == 2
    assert table_segments[1].metadata["data_row_end"] == 3
    assert "table_summary" in table_segments[0].metadata
    assert "chunk_summary" in table_segments[0].metadata
    assert table_segments[0].metadata["table_summary"] == table_segments[1].metadata["table_summary"]


def test_html_splitter_html_output_mode_emits_html_segments() -> None:
    html_text = """
    <html><body>
      <h1>Main</h1>
      <p>Intro</p>
      <table>
        <tr><th>ID</th><th>Value</th></tr>
        <tr><td>1</td><td>A</td></tr>
      </table>
    </body></html>
    """
    splitter = HTMLSplitter(output_format="html")

    segments = splitter.create_segments(html_text)
    table_segments = [segment for segment in segments if segment.type == SegmentType.TABLE]
    text_segments = [segment for segment in segments if segment.type == SegmentType.TEXT]

    assert len(table_segments) == 1
    assert len(text_segments) == 1
    assert text_segments[0].original_format == "html"
    assert "<h1>Main</h1>" in text_segments[0].content
    assert table_segments[0].original_format == "html"
    assert "<table>" in table_segments[0].content
    assert table_segments[0].metadata["output_format"] == "html"


def test_html_splitter_raises_on_malformed_html() -> None:
    splitter = HTMLSplitter()
    with pytest.raises(etree.XMLSyntaxError):
        splitter.create_segments("<html><body><div></span></body></html>")


def test_html_splitter_raises_when_summarizer_fails() -> None:
    html_text = """
    <html><body>
      <table>
        <tr><th>ID</th><th>Value</th></tr>
        <tr><td>1</td><td>A</td></tr>
      </table>
    </body></html>
    """
    splitter = HTMLSplitter(
        summarizer=_FailingSummarizer(),
        summarize_table=True,
    )
    with pytest.raises(RuntimeError):
        splitter.create_segments(html_text)


def test_html_splitter_raises_for_invalid_table_block() -> None:
    splitter = HTMLSplitter()
    with pytest.raises(ValueError):
        splitter.create_segments("<html><body><table></table></body></html>")


def test_html_splitter_rejects_unknown_output_format() -> None:
    with pytest.raises(ValueError):
        HTMLSplitter(output_format="xml")
