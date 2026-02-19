from __future__ import annotations

from dataclasses import dataclass
from html import escape
import re
from typing import Iterable, List, Optional

from lxml import etree, html

from rag_lib.chunkers.table_rows import ParsedTable, render_markdown_table


_STRIP_TAGS = ("script", "style", "noscript", "template")
_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "table",
    "pre",
    "blockquote",
}


@dataclass(frozen=True)
class HTMLBlock:
    kind: str
    markdown: str
    html: str
    heading_level: Optional[int] = None
    heading_title: Optional[str] = None
    parsed_table: Optional[ParsedTable] = None


def parse_html_document(content: str | bytes) -> etree._Element:
    parser = etree.XMLParser(recover=False)
    normalized = _normalize_doctype(content)
    return etree.fromstring(normalized, parser=parser)


def strip_non_content_nodes(document: etree._Element) -> None:
    etree.strip_elements(document, *_STRIP_TAGS, with_tail=False)


def serialize_html_document(document: etree._Element) -> str:
    return etree.tostring(document, encoding="unicode", method="xml").strip()


def extract_structural_blocks(document: etree._Element) -> List[HTMLBlock]:
    body = document.find("body")
    root = body if body is not None else document
    return _extract_children_blocks(root)


def render_blocks_as_markdown(blocks: Iterable[HTMLBlock]) -> str:
    parts = [block.markdown.strip() for block in blocks if block.markdown.strip()]
    return "\n\n".join(parts).strip()


def render_blocks_as_html(blocks: Iterable[HTMLBlock]) -> str:
    parts = [block.html.strip() for block in blocks if block.html.strip()]
    return "\n\n".join(parts).strip()


def parse_html_table_element(
    table_element: etree._Element,
    *,
    use_first_row_as_header: bool = True,
) -> ParsedTable:
    rows: List[List[str]] = []
    for tr in table_element.xpath(".//tr"):
        cells = tr.xpath("./th|./td")
        if not cells:
            continue
        row = [_normalize_whitespace("".join(cell.itertext())) for cell in cells]
        rows.append(row)

    if not rows:
        raise ValueError("HTML table has no rows.")

    width = max(len(row) for row in rows)
    if width <= 0:
        raise ValueError("HTML table has no columns.")

    normalized_rows = [_normalize_row(row, width) for row in rows]
    if use_first_row_as_header:
        header = normalized_rows[0]
        data_rows = normalized_rows[1:]
        return ParsedTable(header=header, rows=data_rows)

    header = [f"Column{i}" for i in range(1, width + 1)]
    return ParsedTable(header=header, rows=normalized_rows)


def render_html_table(header: List[str], rows: List[List[str]]) -> str:
    if not header:
        raise ValueError("HTML table rendering requires a non-empty header.")

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
    ]
    for cell in header:
        lines.append(f"      <th>{escape(cell)}</th>")
    lines.extend(
        [
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )

    width = len(header)
    for row in rows:
        normalized = _normalize_row(row, width)
        lines.append("    <tr>")
        for cell in normalized:
            lines.append(f"      <td>{escape(cell)}</td>")
        lines.append("    </tr>")

    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def _extract_children_blocks(parent: etree._Element) -> List[HTMLBlock]:
    blocks: List[HTMLBlock] = []

    if parent.text and parent.text.strip():
        blocks.append(_text_node_block(parent.text))

    for child in _iter_element_children(parent):
        blocks.extend(_extract_blocks_from_element(child))
        if child.tail and child.tail.strip():
            blocks.append(_text_node_block(child.tail))

    return blocks


def _extract_blocks_from_element(element: etree._Element) -> List[HTMLBlock]:
    tag = _local_name(element.tag)

    if tag in _BLOCK_TAGS:
        block = _build_block(element, tag)
        return [block]

    # Container or unknown tags recurse into children to avoid flattening hierarchy.
    return _extract_children_blocks(element)


def _build_block(element: etree._Element, tag: str) -> HTMLBlock:
    if tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
        level = int(tag[1])
        title = _normalize_whitespace("".join(element.itertext()))
        if not title:
            raise ValueError("Heading block has no title.")
        return HTMLBlock(
            kind="heading",
            markdown=f"{'#' * level} {title}",
            html=_serialize_element(element),
            heading_level=level,
            heading_title=title,
        )

    if tag in {"ul", "ol"}:
        markdown = _render_list_markdown(element, depth=0)
        if not markdown.strip():
            raise ValueError("List block is empty.")
        return HTMLBlock(kind="list", markdown=markdown, html=_serialize_element(element))

    if tag == "table":
        parsed = parse_html_table_element(element, use_first_row_as_header=True)
        markdown = render_markdown_table(parsed.header, parsed.rows)
        return HTMLBlock(
            kind="table",
            markdown=markdown,
            html=_serialize_element(element),
            parsed_table=parsed,
        )

    if tag == "pre":
        code = "".join(element.itertext()).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        markdown = f"```\n{code}\n```"
        return HTMLBlock(kind="code", markdown=markdown, html=_serialize_element(element))

    if tag == "blockquote":
        text = _render_inline_markdown(element).strip()
        lines = text.splitlines() or [text]
        markdown = "\n".join("> " + line if line else ">" for line in lines)
        return HTMLBlock(kind="blockquote", markdown=markdown, html=_serialize_element(element))

    # Default text-like block (p and any future additions routed here).
    text_markdown = _render_inline_markdown(element).strip()
    return HTMLBlock(kind="text", markdown=text_markdown, html=_serialize_element(element))


def _render_list_markdown(list_element: etree._Element, depth: int) -> str:
    ordered = _local_name(list_element.tag) == "ol"
    lines: List[str] = []
    index = 1

    for li in list_element.xpath("./li"):
        marker = f"{index}." if ordered else "-"
        item_text = _render_list_item_text(li).strip()
        prefix = "  " * depth + marker + " "
        lines.append(prefix + item_text if item_text else prefix.rstrip())

        for nested in li.xpath("./ul|./ol"):
            nested_markdown = _render_list_markdown(nested, depth + 1)
            if nested_markdown:
                lines.extend(nested_markdown.splitlines())

        index += 1

    return "\n".join(lines)


def _render_list_item_text(li: etree._Element) -> str:
    parts: List[str] = []
    if li.text:
        parts.append(_escape_markdown_text(li.text))

    for child in _iter_element_children(li):
        tag = _local_name(child.tag)
        if tag in {"ul", "ol"}:
            if child.tail:
                parts.append(_escape_markdown_text(child.tail))
            continue
        parts.append(_render_inline_markdown(child))
        if child.tail:
            parts.append(_escape_markdown_text(child.tail))

    return _normalize_whitespace("".join(parts))


def _render_inline_markdown(element: etree._Element) -> str:
    parts: List[str] = []
    if element.text:
        parts.append(_escape_markdown_text(element.text))

    for child in _iter_element_children(element):
        tag = _local_name(child.tag)
        child_rendered = _render_inline_markdown(child)

        if tag == "br":
            child_rendered = "\n"
        elif tag in {"strong", "b"}:
            child_rendered = f"**{child_rendered}**"
        elif tag in {"em", "i"}:
            child_rendered = f"*{child_rendered}*"
        elif tag == "code":
            child_rendered = f"`{child_rendered}`"
        elif tag == "a":
            href = child.get("href", "").strip()
            child_rendered = f"[{child_rendered}]({href})" if href else child_rendered
        elif tag == "img":
            alt = child.get("alt", "image")
            src = child.get("src", "").strip()
            child_rendered = f"![{alt}]({src})" if src else f"![{alt}]()"

        parts.append(child_rendered)
        if child.tail:
            parts.append(_escape_markdown_text(child.tail))

    return "".join(parts)


def _text_node_block(text: str) -> HTMLBlock:
    normalized = _normalize_whitespace(text)
    return HTMLBlock(
        kind="text",
        markdown=_escape_markdown_text(normalized),
        html=f"<p>{escape(normalized)}</p>",
    )


def _serialize_element(element: etree._Element) -> str:
    return html.tostring(element, encoding="unicode", method="html", with_tail=False).strip()


def _escape_markdown_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
    )


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def _normalize_row(row: List[str], width: int) -> List[str]:
    if len(row) == width:
        return row
    if len(row) > width:
        return row[:width]
    return row + [""] * (width - len(row))


def _iter_element_children(element: etree._Element) -> Iterable[etree._Element]:
    for child in element:
        if isinstance(child.tag, str):
            yield child


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1].lower()
    return tag.lower()


def _normalize_doctype(content: str | bytes) -> bytes:
    if isinstance(content, bytes):
        text = content.decode("utf-8")
    else:
        text = content
    normalized = re.sub(
        pattern=r"(?is)<!doctype\s+html>",
        repl="<!DOCTYPE html>",
        string=text,
        count=1,
    )
    return normalized.encode("utf-8")
