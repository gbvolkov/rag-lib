from pathlib import Path

import pytest
from lxml import etree

from rag_lib.loaders.html import HTMLLoader


def test_html_loader_markdown_mode_returns_one_document(tmp_path: Path) -> None:
    html_path = tmp_path / "sample.html"
    html_path.write_text(
        """
        <html>
          <body>
            <h1>Title</h1>
            <p>Hello <a href="https://example.com">world</a>.</p>
            <ul><li>One</li><li>Two</li></ul>
            <table>
              <tr><th>Name</th><th>Value</th></tr>
              <tr><td>A</td><td>1</td></tr>
            </table>
            <script>console.log('x')</script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    docs = HTMLLoader(str(html_path), output_format="markdown").load()

    assert len(docs) == 1
    doc = docs[0]
    assert "# Title" in doc.page_content
    assert "[world](https://example.com)" in doc.page_content
    assert "- One" in doc.page_content
    assert "| Name | Value |" in doc.page_content
    assert "console.log" not in doc.page_content
    assert doc.metadata["source"] == str(html_path)
    assert doc.metadata["source_type"] == "html"
    assert doc.metadata["output_format"] == "markdown"


def test_html_loader_html_mode_strips_non_content_nodes(tmp_path: Path) -> None:
    html_path = tmp_path / "sample.html"
    html_path.write_text(
        """
        <html>
          <body>
            <h1>Title</h1>
            <p>Body</p>
            <style>body { color: red; }</style>
            <noscript>fallback</noscript>
            <script>console.log('x')</script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    docs = HTMLLoader(str(html_path), output_format="html").load()

    assert len(docs) == 1
    content = docs[0].page_content
    assert "<h1>Title</h1>" in content
    assert "<p>Body</p>" in content
    assert "<script" not in content
    assert "<style" not in content
    assert "<noscript" not in content
    assert docs[0].metadata["output_format"] == "html"


def test_html_loader_rejects_unknown_output_format() -> None:
    with pytest.raises(ValueError):
        HTMLLoader("dummy.html", output_format="xml")


def test_html_loader_raises_on_malformed_html(tmp_path: Path) -> None:
    html_path = tmp_path / "broken.html"
    html_path.write_text("<html><body><div></span></body></html>", encoding="utf-8")

    loader = HTMLLoader(str(html_path), output_format="markdown")
    with pytest.raises(etree.XMLSyntaxError):
        loader.load()
