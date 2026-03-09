from __future__ import annotations

import mimetypes
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag_lib.core.logger import logger
from rag_lib.summarizers.presentation_visual import (
    LLMPresentationVisualSummarizer,
    PresentationVisual,
    PresentationVisualSummarizer,
)

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
DGM_NS = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

NS = {
    "p": P_NS,
    "a": A_NS,
    "r": R_NS,
    "c": C_NS,
    "dgm": DGM_NS,
    "ct": CT_NS,
}

TITLE_PLACEHOLDER_TYPES = {"title", "ctrTitle"}
NOTES_SKIPPED_PLACEHOLDER_TYPES = {"dt", "ftr", "hdr", "sldImg", "sldNum"}


@dataclass(frozen=True)
class _Relationship:
    target: str
    rel_type: str
    target_mode: Optional[str] = None


@dataclass
class _ContentTypes:
    defaults: Dict[str, str]
    overrides: Dict[str, str]

    def resolve(self, part_name: str) -> str:
        normalized = part_name.lstrip("/")
        if normalized in self.overrides:
            return self.overrides[normalized]
        extension = Path(normalized).suffix.lower().lstrip(".")
        return self.defaults.get(extension, "")


@dataclass
class _SlideBlock:
    kind: Literal["text", "table", "image", "chart", "smartart"]
    markdown: str
    placeholder_type: Optional[str] = None
    shape_name: str = ""
    mime_type: str = ""
    image_bytes: bytes | None = None
    structured_markdown: str = ""


@dataclass(frozen=True)
class _ListInfo:
    kind: Optional[Literal["bullet", "ordered"]]
    level: int
    start_at: int = 1


class PPTXLoader:
    """
    Strict PPTX -> Markdown loader preserving slide order and major structure.
    """

    def __init__(
        self,
        file_path: str,
        *,
        include_notes: bool = True,
        summarize_visuals: bool = False,
        visual_summarizer: PresentationVisualSummarizer | None = None,
        llm: BaseChatModel | None = None,
    ):
        self.file_path = file_path
        self.include_notes = include_notes
        self.summarize_visuals = summarize_visuals
        self.visual_summarizer = visual_summarizer
        self.llm = llm

    def load(self) -> List[Document]:
        logger.info(
            "Loading PPTX: %s include_notes=%s summarize_visuals=%s",
            self._safe_log_text(self.file_path),
            self.include_notes,
            self.summarize_visuals,
        )

        file_path = Path(self.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            with zipfile.ZipFile(file_path, "r") as archive:
                content_types = self._read_content_types(archive)
                slide_parts = self._read_slide_part_names(archive)
                if not slide_parts:
                    raise RuntimeError("Presentation does not contain any slides.")

                summarizer = self._resolve_visual_summarizer()
                slides_markdown = [
                    self._render_slide(
                        archive=archive,
                        slide_part=slide_part,
                        slide_index=slide_index,
                        content_types=content_types,
                        visual_summarizer=summarizer,
                    )
                    for slide_index, slide_part in enumerate(slide_parts, start=1)
                ]
        except FileNotFoundError:
            raise
        except RuntimeError:
            raise
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Invalid PPTX file: {self.file_path}") from exc
        except Exception as exc:
            logger.error(
                "Failed to load PPTX '%s': %s",
                self._safe_log_text(self.file_path),
                self._safe_log_text(str(exc)),
            )
            raise RuntimeError(f"Failed to load PPTX: {exc}") from exc

        markdown = "\n\n".join(part for part in slides_markdown if part.strip()).strip()
        metadata = {
            "source": self.file_path,
            "source_type": "pptx",
            "output_format": "markdown",
            "slide_count": len(slide_parts),
            "include_notes": self.include_notes,
            "summarize_visuals": self.summarize_visuals,
        }
        return [Document(page_content=markdown, metadata=metadata)]

    def _resolve_visual_summarizer(self) -> PresentationVisualSummarizer | None:
        if not self.summarize_visuals:
            return None
        if self.visual_summarizer is not None:
            return self.visual_summarizer
        return LLMPresentationVisualSummarizer(llm=self.llm)

    def _read_content_types(self, archive: zipfile.ZipFile) -> _ContentTypes:
        root = self._read_xml_part(archive, "[Content_Types].xml", required=True)
        defaults: Dict[str, str] = {}
        overrides: Dict[str, str] = {}
        for node in root.findall("ct:Default", NS):
            extension = (node.get("Extension") or "").lower()
            content_type = node.get("ContentType", "")
            if extension:
                defaults[extension] = content_type
        for node in root.findall("ct:Override", NS):
            part_name = (node.get("PartName") or "").lstrip("/")
            content_type = node.get("ContentType", "")
            if part_name:
                overrides[part_name] = content_type
        return _ContentTypes(defaults=defaults, overrides=overrides)

    def _read_slide_part_names(self, archive: zipfile.ZipFile) -> List[str]:
        presentation_part = "ppt/presentation.xml"
        presentation_root = self._read_xml_part(archive, presentation_part, required=True)
        relationships = self._read_relationships(archive, presentation_part)
        slide_parts: List[str] = []
        slide_id_list = presentation_root.find("p:sldIdLst", NS)
        if slide_id_list is None:
            raise RuntimeError("Presentation is missing p:sldIdLst.")

        for slide_id in slide_id_list.findall("p:sldId", NS):
            rel_id = slide_id.get(f"{{{R_NS}}}id")
            if not rel_id or rel_id not in relationships:
                raise RuntimeError("Presentation slide relationship is missing.")
            relationship = relationships[rel_id]
            slide_parts.append(self._resolve_part_name(presentation_part, relationship.target))
        return slide_parts

    def _render_slide(
        self,
        *,
        archive: zipfile.ZipFile,
        slide_part: str,
        slide_index: int,
        content_types: _ContentTypes,
        visual_summarizer: PresentationVisualSummarizer | None,
    ) -> str:
        slide_root = self._read_xml_part(archive, slide_part, required=True)
        relationships = self._read_relationships(archive, slide_part)
        blocks = self._parse_slide_blocks(
            archive=archive,
            slide_root=slide_root,
            slide_part=slide_part,
            relationships=relationships,
            slide_index=slide_index,
            content_types=content_types,
        )

        title, title_block_index, remainder = self._resolve_slide_title(blocks, slide_index)
        if title_block_index is not None:
            if remainder:
                blocks[title_block_index].markdown = remainder
            else:
                blocks[title_block_index].markdown = ""

        if visual_summarizer is not None:
            self._summarize_visual_blocks(
                blocks=blocks,
                slide_index=slide_index,
                slide_title=title,
                summarizer=visual_summarizer,
            )

        parts: List[str] = [f"# Slide {slide_index}: {title}"]
        body_blocks = [block.markdown.strip() for block in blocks if block.markdown.strip()]
        if body_blocks:
            parts.append("\n\n".join(body_blocks))

        notes_markdown = ""
        if self.include_notes:
            notes_markdown = self._extract_notes_markdown(
                archive=archive,
                slide_part=slide_part,
                slide_relationships=relationships,
            )
        if notes_markdown:
            parts.append(f"### Speaker Notes\n\n{notes_markdown}")

        return "\n\n".join(part for part in parts if part.strip()).strip()

    def _parse_slide_blocks(
        self,
        *,
        archive: zipfile.ZipFile,
        slide_root: ET.Element,
        slide_part: str,
        relationships: Dict[str, _Relationship],
        slide_index: int,
        content_types: _ContentTypes,
    ) -> List[_SlideBlock]:
        sp_tree = slide_root.find("p:cSld/p:spTree", NS)
        if sp_tree is None:
            raise RuntimeError(f"Slide {slide_index} is missing p:spTree.")

        blocks: List[_SlideBlock] = []
        for node in self._iter_shape_nodes(sp_tree):
            block = self._parse_shape_block(
                archive=archive,
                node=node,
                slide_part=slide_part,
                relationships=relationships,
                content_types=content_types,
            )
            if block is not None:
                blocks.append(block)
        return blocks

    def _iter_shape_nodes(self, container: ET.Element) -> Iterable[ET.Element]:
        for child in list(container):
            local_name = _local_name(child.tag)
            if local_name == "grpSp":
                yield from self._iter_shape_nodes(child)
            elif local_name in {"sp", "pic", "graphicFrame"}:
                yield child

    def _parse_shape_block(
        self,
        *,
        archive: zipfile.ZipFile,
        node: ET.Element,
        slide_part: str,
        relationships: Dict[str, _Relationship],
        content_types: _ContentTypes,
    ) -> _SlideBlock | None:
        local_name = _local_name(node.tag)
        if local_name == "sp":
            return self._parse_text_shape(node=node, relationships=relationships, in_table=False)
        if local_name == "pic":
            return self._parse_picture_block(
                archive=archive,
                picture=node,
                slide_part=slide_part,
                relationships=relationships,
                content_types=content_types,
            )
        if local_name == "graphicFrame":
            table = node.find(".//a:tbl", NS)
            if table is not None:
                return self._parse_table_block(node=node, table=table, relationships=relationships)

            chart_ref = node.find(".//c:chart", NS)
            if chart_ref is not None:
                return self._parse_chart_block(
                    archive=archive,
                    graphic_frame=node,
                    chart_ref=chart_ref,
                    slide_part=slide_part,
                    relationships=relationships,
                )

            rel_ids = node.find(".//dgm:relIds", NS)
            if rel_ids is not None and rel_ids.get(f"{{{R_NS}}}dm"):
                return self._parse_smartart_block(
                    archive=archive,
                    graphic_frame=node,
                    rel_ids=rel_ids,
                    slide_part=slide_part,
                    relationships=relationships,
                )
        return None

    def _parse_text_shape(
        self,
        *,
        node: ET.Element,
        relationships: Dict[str, _Relationship],
        in_table: bool,
    ) -> _SlideBlock | None:
        text_body = node.find("p:txBody", NS)
        if text_body is None:
            return None

        paragraphs = text_body.findall("a:p", NS)
        list_counters: Dict[int, int] = {}
        rendered: List[str] = []
        for paragraph in paragraphs:
            markdown = self._render_text_paragraph(
                paragraph=paragraph,
                relationships=relationships,
                in_table=in_table,
                list_counters=list_counters,
            )
            if markdown:
                rendered.append(markdown)

        markdown = "\n\n".join(part for part in rendered if part.strip()).strip()
        if not markdown:
            return None

        placeholder = node.find("p:nvSpPr/p:nvPr/p:ph", NS)
        placeholder_type = placeholder.get("type") if placeholder is not None else None
        c_nv_pr = node.find("p:nvSpPr/p:cNvPr", NS)
        shape_name = c_nv_pr.get("name", "") if c_nv_pr is not None else ""

        return _SlideBlock(
            kind="text",
            markdown=markdown,
            placeholder_type=placeholder_type,
            shape_name=shape_name,
        )

    def _render_text_paragraph(
        self,
        *,
        paragraph: ET.Element,
        relationships: Dict[str, _Relationship],
        in_table: bool,
        list_counters: Dict[int, int],
    ) -> str:
        list_info = self._read_list_info(paragraph)
        text = self._render_paragraph_runs(paragraph=paragraph, relationships=relationships, in_table=in_table)
        text = "\n".join(line.rstrip() for line in text.splitlines())
        if not text.strip():
            return ""

        if list_info.kind is None:
            list_counters.clear()
            return text.strip()

        for level in list(list_counters.keys()):
            if level > list_info.level:
                del list_counters[level]

        indent = "  " * max(0, list_info.level)
        if list_info.kind == "ordered":
            counter = list_counters.get(list_info.level, list_info.start_at - 1) + 1
            list_counters[list_info.level] = counter
            marker = f"{counter}."
        else:
            marker = "-"

        continuation_indent = " " * (len(indent) + len(marker) + 1)
        lines = text.strip().split("\n")
        first_line = f"{indent}{marker} {lines[0].strip()}"
        tail = [f"{continuation_indent}{line.strip()}" for line in lines[1:] if line.strip()]
        return "\n".join([first_line, *tail]).rstrip()

    def _render_paragraph_runs(
        self,
        *,
        paragraph: ET.Element,
        relationships: Dict[str, _Relationship],
        in_table: bool,
    ) -> str:
        parts: List[str] = []
        for child in list(paragraph):
            local_name = _local_name(child.tag)
            if local_name in {"r", "fld"}:
                text = "".join(t.text or "" for t in child.findall(".//a:t", NS))
                if not text:
                    continue
                rpr = child.find("a:rPr", NS)
                parts.append(
                    self._format_text_run(
                        text=text,
                        rpr=rpr,
                        relationships=relationships,
                        in_table=in_table,
                    )
                )
            elif local_name == "br":
                parts.append("\n")
            elif local_name == "tab":
                parts.append("\t")

        return "".join(parts).strip()

    def _format_text_run(
        self,
        *,
        text: str,
        rpr: ET.Element | None,
        relationships: Dict[str, _Relationship],
        in_table: bool,
    ) -> str:
        escaped = self._escape_markdown(text, in_table=in_table)
        bold = self._attr_truthy(rpr, "b")
        italic = self._attr_truthy(rpr, "i")
        underline = self._attr_present_and_not_none(rpr, "u")
        strike = self._attr_present_and_not_none(rpr, "strike")

        formatted = self._apply_formatting(
            text=escaped,
            bold=bold,
            italic=italic,
            underline=underline,
            strike=strike,
        )

        hyperlink = self._hyperlink_target(rpr, relationships)
        if hyperlink:
            return f"[{formatted}]({hyperlink})"
        return formatted

    def _parse_picture_block(
        self,
        *,
        archive: zipfile.ZipFile,
        picture: ET.Element,
        slide_part: str,
        relationships: Dict[str, _Relationship],
        content_types: _ContentTypes,
    ) -> _SlideBlock | None:
        c_nv_pr = picture.find("p:nvPicPr/p:cNvPr", NS)
        shape_name = ""
        description = ""
        if c_nv_pr is not None:
            shape_name = c_nv_pr.get("name", "")
            description = c_nv_pr.get("descr", "")

        blip = picture.find(".//a:blip", NS)
        rel_id = blip.get(f"{{{R_NS}}}embed") if blip is not None else None
        if not rel_id or rel_id not in relationships:
            return None

        relationship = relationships[rel_id]
        image_part = self._resolve_part_name(slide_part, relationship.target)
        image_bytes = self._read_bytes_part(archive, image_part, required=True)
        mime_type = content_types.resolve(image_part) or mimetypes.guess_type(image_part)[0] or "image/png"
        label = shape_name or Path(image_part).name
        placeholder = f"[image: {label}]"
        structured_markdown = description.strip()

        return _SlideBlock(
            kind="image",
            markdown=placeholder,
            shape_name=label,
            mime_type=mime_type,
            image_bytes=image_bytes,
            structured_markdown=structured_markdown,
        )

    def _parse_table_block(
        self,
        *,
        node: ET.Element,
        table: ET.Element,
        relationships: Dict[str, _Relationship],
    ) -> _SlideBlock | None:
        rows: List[List[str]] = []
        for row in table.findall("a:tr", NS):
            cells: List[str] = []
            for cell in row.findall("a:tc", NS):
                cells.append(self._render_table_cell(cell, relationships))
            if cells:
                rows.append(cells)

        markdown = self._render_markdown_table(rows)
        if not markdown:
            return None

        c_nv_pr = node.find("p:nvGraphicFramePr/p:cNvPr", NS)
        shape_name = c_nv_pr.get("name", "") if c_nv_pr is not None else ""
        return _SlideBlock(kind="table", markdown=markdown, shape_name=shape_name)

    def _render_table_cell(self, cell: ET.Element, relationships: Dict[str, _Relationship]) -> str:
        paragraphs = cell.findall("a:txBody/a:p", NS)
        list_counters: Dict[int, int] = {}
        rendered: List[str] = []
        for paragraph in paragraphs:
            markdown = self._render_text_paragraph(
                paragraph=paragraph,
                relationships=relationships,
                in_table=True,
                list_counters=list_counters,
            )
            if markdown:
                rendered.append(markdown)
        return "<br>".join(part.replace("|", "\\|") for part in rendered if part.strip())

    def _parse_chart_block(
        self,
        *,
        archive: zipfile.ZipFile,
        graphic_frame: ET.Element,
        chart_ref: ET.Element,
        slide_part: str,
        relationships: Dict[str, _Relationship],
    ) -> _SlideBlock | None:
        rel_id = chart_ref.get(f"{{{R_NS}}}id")
        if not rel_id or rel_id not in relationships:
            raise RuntimeError("Chart relationship is missing from slide.")
        chart_part = self._resolve_part_name(slide_part, relationships[rel_id].target)
        chart_root = self._read_xml_part(archive, chart_part, required=True)

        c_nv_pr = graphic_frame.find("p:nvGraphicFramePr/p:cNvPr", NS)
        shape_name = c_nv_pr.get("name", "") if c_nv_pr is not None else ""
        chart_title, structured_markdown = self._render_chart_content(chart_root)
        label = chart_title or shape_name or f"Chart {Path(chart_part).stem}"
        parts = [f"[chart: {label}]"]
        if structured_markdown:
            parts.append(structured_markdown)
        markdown = "\n\n".join(parts)

        return _SlideBlock(
            kind="chart",
            markdown=markdown,
            shape_name=label,
            mime_type="application/vnd.openxmlformats-officedocument.drawingml.chart+xml",
            structured_markdown=structured_markdown,
        )

    def _render_chart_content(self, chart_root: ET.Element) -> tuple[str, str]:
        title = self._extract_chart_title(chart_root)
        series_nodes = chart_root.findall(".//c:plotArea/*/c:ser", NS)
        series_payloads: List[tuple[str, List[str], List[str]]] = []
        for series_index, series in enumerate(series_nodes, start=1):
            series_name = self._extract_chart_series_name(series) or f"Series {series_index}"
            category_node = series.find("c:cat", NS)
            if category_node is None:
                category_node = series.find("c:xVal", NS)
            value_node = series.find("c:val", NS)
            if value_node is None:
                value_node = series.find("c:yVal", NS)
            categories = self._extract_chart_values(category_node)
            values = self._extract_chart_values(value_node)
            if not categories:
                categories = [f"Item {i}" for i in range(1, len(values) + 1)]
            series_payloads.append((series_name, categories, values))

        structured_parts: List[str] = []
        if title:
            structured_parts.append(f"Title: {title}")

        table_markdown = self._render_chart_table(series_payloads)
        if table_markdown:
            structured_parts.append(table_markdown)
        elif series_payloads:
            bullets = []
            for series_name, categories, values in series_payloads:
                pairs = ", ".join(
                    f"{categories[idx]}={values[idx]}"
                    for idx in range(min(len(categories), len(values)))
                )
                bullets.append(f"- {series_name}: {pairs}".rstrip(": "))
            structured_parts.append("\n".join(bullets))

        return title, "\n\n".join(part for part in structured_parts if part.strip()).strip()

    def _render_chart_table(self, series_payloads: List[tuple[str, List[str], List[str]]]) -> str:
        if not series_payloads:
            return ""

        max_len = max(max(len(categories), len(values)) for _, categories, values in series_payloads)
        if max_len <= 0:
            return ""

        row_labels: List[str] = []
        for row_index in range(max_len):
            label = ""
            for _, categories, _ in series_payloads:
                if row_index < len(categories) and categories[row_index]:
                    label = categories[row_index]
                    break
            if not label:
                label = f"Item {row_index + 1}"
            row_labels.append(label)

        rows: List[List[str]] = []
        header = ["Category", *[series_name for series_name, _, _ in series_payloads]]
        rows.append(header)
        for row_index, label in enumerate(row_labels):
            row = [label]
            for _, _, values in series_payloads:
                row.append(values[row_index] if row_index < len(values) else "")
            rows.append(row)
        return self._render_markdown_table(rows)

    def _parse_smartart_block(
        self,
        *,
        archive: zipfile.ZipFile,
        graphic_frame: ET.Element,
        rel_ids: ET.Element,
        slide_part: str,
        relationships: Dict[str, _Relationship],
    ) -> _SlideBlock | None:
        rel_id = rel_ids.get(f"{{{R_NS}}}dm")
        if not rel_id or rel_id not in relationships:
            raise RuntimeError("SmartArt data relationship is missing from slide.")
        data_part = self._resolve_part_name(slide_part, relationships[rel_id].target)
        data_root = self._read_xml_part(archive, data_part, required=True)

        c_nv_pr = graphic_frame.find("p:nvGraphicFramePr/p:cNvPr", NS)
        shape_name = c_nv_pr.get("name", "") if c_nv_pr is not None else ""
        structured_markdown = self._render_smartart_content(data_root)
        label = shape_name or f"SmartArt {Path(data_part).stem}"
        parts = [f"[smartart: {label}]"]
        if structured_markdown:
            parts.append(structured_markdown)
        markdown = "\n\n".join(parts)

        return _SlideBlock(
            kind="smartart",
            markdown=markdown,
            shape_name=label,
            mime_type="application/vnd.openxmlformats-officedocument.drawingml.diagramData+xml",
            structured_markdown=structured_markdown,
        )

    def _render_smartart_content(self, data_root: ET.Element) -> str:
        points: Dict[str, str] = {}
        point_order: List[str] = []
        for point in data_root.findall(".//dgm:ptLst/dgm:pt", NS):
            model_id = point.get("modelId", "")
            if not model_id:
                continue
            text = "".join(
                part.text or ""
                for part in point.findall(".//dgm:t", NS) + point.findall(".//a:t", NS)
            ).strip()
            points[model_id] = text
            point_order.append(model_id)

        children: Dict[str, List[str]] = {}
        incoming: set[str] = set()
        for connection in data_root.findall(".//dgm:cxnLst/dgm:cxn", NS):
            src_id = connection.get("srcId", "")
            dest_id = connection.get("destId", "")
            if not src_id or not dest_id:
                continue
            children.setdefault(src_id, []).append(dest_id)
            incoming.add(dest_id)

        roots = [point_id for point_id in point_order if point_id not in incoming]
        if not roots:
            roots = point_order

        lines: List[str] = []
        visited: set[str] = set()

        def walk(node_id: str, depth: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)

            text = points.get(node_id, "").strip()
            if text:
                lines.append(f"{'  ' * depth}- {text}")

            for child_id in children.get(node_id, []):
                walk(child_id, depth + 1)

        for root_id in roots:
            walk(root_id, 0)

        if not lines:
            for point_id in point_order:
                text = points.get(point_id, "").strip()
                if text:
                    lines.append(f"- {text}")

        return "\n".join(lines).strip()

    def _resolve_slide_title(
        self,
        blocks: List[_SlideBlock],
        slide_index: int,
    ) -> tuple[str, int | None, str]:
        candidate_index: int | None = None
        fallback_index: int | None = None

        for index, block in enumerate(blocks):
            if block.kind != "text" or not block.markdown.strip():
                continue
            if block.placeholder_type in TITLE_PLACEHOLDER_TYPES:
                candidate_index = index
                break
            if fallback_index is None:
                fallback_index = index

        chosen_index = candidate_index if candidate_index is not None else fallback_index
        if chosen_index is None:
            return f"Slide {slide_index}", None, ""

        title_line, remainder = self._split_title_block(blocks[chosen_index].markdown)
        if not title_line:
            return f"Slide {slide_index}", chosen_index, blocks[chosen_index].markdown
        return title_line, chosen_index, remainder

    def _split_title_block(self, markdown: str) -> tuple[str, str]:
        lines = markdown.splitlines()
        title = ""
        title_index = -1
        for index, line in enumerate(lines):
            candidate = self._strip_markdown_text(line).strip()
            if candidate:
                title = candidate
                title_index = index
                break

        if title_index < 0:
            return "", markdown.strip()

        remainder_lines = [line for index, line in enumerate(lines) if index != title_index]
        remainder = "\n".join(remainder_lines).strip()
        return title, remainder

    def _summarize_visual_blocks(
        self,
        *,
        blocks: List[_SlideBlock],
        slide_index: int,
        slide_title: str,
        summarizer: PresentationVisualSummarizer,
    ) -> None:
        for block in blocks:
            if block.kind not in {"image", "chart", "smartart"}:
                continue
            visual = PresentationVisual(
                kind=block.kind,
                slide_index=slide_index,
                slide_title=slide_title,
                shape_name=block.shape_name,
                mime_type=block.mime_type,
                image_bytes=block.image_bytes,
                structured_markdown=block.structured_markdown,
            )
            summary = summarizer.summarize(visual).strip()
            if summary:
                block.markdown = f"{block.markdown}\n\nSummary: {summary}".strip()

    def _extract_notes_markdown(
        self,
        *,
        archive: zipfile.ZipFile,
        slide_part: str,
        slide_relationships: Dict[str, _Relationship],
    ) -> str:
        notes_part = ""
        for relationship in slide_relationships.values():
            if relationship.rel_type.endswith("/notesSlide"):
                notes_part = self._resolve_part_name(slide_part, relationship.target)
                break
        if not notes_part:
            return ""

        notes_root = self._read_xml_part(archive, notes_part, required=True)
        sp_tree = notes_root.find("p:cSld/p:spTree", NS)
        if sp_tree is None:
            return ""

        note_relationships = self._read_relationships(archive, notes_part)
        blocks: List[str] = []
        for shape in self._iter_shape_nodes(sp_tree):
            if _local_name(shape.tag) != "sp":
                continue
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            placeholder_type = placeholder.get("type") if placeholder is not None else None
            if placeholder_type in NOTES_SKIPPED_PLACEHOLDER_TYPES:
                continue
            block = self._parse_text_shape(node=shape, relationships=note_relationships, in_table=False)
            if block and block.markdown.strip():
                blocks.append(block.markdown.strip())
        return "\n\n".join(blocks).strip()

    def _read_xml_part(
        self,
        archive: zipfile.ZipFile,
        part_name: str,
        *,
        required: bool,
    ) -> ET.Element:
        raw = self._read_bytes_part(archive, part_name, required=required)
        try:
            return ET.fromstring(raw)
        except ET.ParseError as exc:
            raise RuntimeError(f"Invalid XML in part '{part_name}'.") from exc

    def _read_bytes_part(
        self,
        archive: zipfile.ZipFile,
        part_name: str,
        *,
        required: bool,
    ) -> bytes:
        try:
            return archive.read(part_name)
        except KeyError as exc:
            if required:
                raise RuntimeError(f"Required PPTX part is missing: {part_name}") from exc
            return b""

    def _read_relationships(
        self,
        archive: zipfile.ZipFile,
        part_name: str,
    ) -> Dict[str, _Relationship]:
        rels_part = self._relationship_part_name(part_name)
        raw = self._read_bytes_part(archive, rels_part, required=False)
        if not raw:
            return {}
        try:
            rels_root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise RuntimeError(f"Invalid XML in part '{rels_part}'.") from exc

        relationships: Dict[str, _Relationship] = {}
        for relationship in rels_root.findall(f"{{{PKG_REL_NS}}}Relationship"):
            rel_id = relationship.get("Id")
            target = relationship.get("Target")
            if not rel_id or not target:
                continue
            relationships[rel_id] = _Relationship(
                target=target,
                rel_type=relationship.get("Type", ""),
                target_mode=relationship.get("TargetMode"),
            )
        return relationships

    @staticmethod
    def _relationship_part_name(part_name: str) -> str:
        directory = posixpath.dirname(part_name)
        filename = posixpath.basename(part_name)
        rels_dir = posixpath.join(directory, "_rels") if directory else "_rels"
        return posixpath.join(rels_dir, f"{filename}.rels")

    @staticmethod
    def _resolve_part_name(base_part: str, target: str) -> str:
        return posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))

    def _extract_chart_title(self, chart_root: ET.Element) -> str:
        title_parts = [
            text.text or ""
            for text in chart_root.findall(".//c:title//a:t", NS)
        ]
        return "".join(title_parts).strip()

    def _extract_chart_series_name(self, series: ET.Element) -> str:
        series_name = [
            text.text or ""
            for text in series.findall("c:tx//c:v", NS) + series.findall("c:tx//a:t", NS)
        ]
        return "".join(series_name).strip()

    def _extract_chart_values(self, node: ET.Element | None) -> List[str]:
        if node is None:
            return []
        values = [
            (value.text or "").strip()
            for value in node.findall(".//c:pt/c:v", NS)
            if (value.text or "").strip()
        ]
        if values:
            return values
        return [
            (value.text or "").strip()
            for value in node.findall(".//c:v", NS)
            if (value.text or "").strip()
        ]

    def _read_list_info(self, paragraph: ET.Element) -> _ListInfo:
        ppr = paragraph.find("a:pPr", NS)
        if ppr is None:
            return _ListInfo(kind=None, level=0)

        try:
            level = int(ppr.get("lvl", "0"))
        except ValueError:
            level = 0

        if ppr.find("a:buNone", NS) is not None:
            return _ListInfo(kind=None, level=level)

        auto_num = ppr.find("a:buAutoNum", NS)
        if auto_num is not None:
            try:
                start_at = int(auto_num.get("startAt", "1"))
            except ValueError:
                start_at = 1
            return _ListInfo(kind="ordered", level=level, start_at=start_at)

        if ppr.find("a:buChar", NS) is not None or ppr.find("a:buBlip", NS) is not None:
            return _ListInfo(kind="bullet", level=level)

        if ppr.get("lvl") is not None:
            return _ListInfo(kind="bullet", level=level)

        return _ListInfo(kind=None, level=level)

    def _hyperlink_target(
        self,
        rpr: ET.Element | None,
        relationships: Dict[str, _Relationship],
    ) -> str:
        if rpr is None:
            return ""
        hyperlink = rpr.find("a:hlinkClick", NS)
        if hyperlink is None:
            return ""
        rel_id = hyperlink.get(f"{{{R_NS}}}id")
        if not rel_id or rel_id not in relationships:
            return ""
        relationship = relationships[rel_id]
        if relationship.target_mode == "External":
            return relationship.target
        match = re.search(r"slide(\d+)\.xml$", relationship.target)
        if match:
            return f"#slide-{match.group(1)}"
        return relationship.target

    @staticmethod
    def _attr_truthy(rpr: ET.Element | None, name: str) -> bool:
        if rpr is None:
            return False
        value = rpr.get(name)
        return value in {"1", "true", "True"}

    @staticmethod
    def _attr_present_and_not_none(rpr: ET.Element | None, name: str) -> bool:
        if rpr is None:
            return False
        value = rpr.get(name)
        return value not in {None, "", "none", "noStrike"}

    def _render_markdown_table(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (width - len(row)) for row in rows]
        header = normalized_rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _apply_formatting(
        self,
        *,
        text: str,
        bold: bool,
        italic: bool,
        underline: bool,
        strike: bool,
    ) -> str:
        if not text:
            return text

        parts = re.split(r"(\n)", text)
        output: List[str] = []
        for part in parts:
            if part == "\n":
                output.append(part)
                continue
            if not part or part.isspace():
                output.append(part)
                continue

            formatted = part
            if bold and italic:
                formatted = f"***{formatted}***"
            elif bold:
                formatted = f"**{formatted}**"
            elif italic:
                formatted = f"*{formatted}*"

            if strike:
                formatted = f"~~{formatted}~~"
            if underline:
                formatted = f"<u>{formatted}</u>"
            output.append(formatted)
        return "".join(output)

    def _escape_markdown(self, text: str, *, in_table: bool) -> str:
        escaped = text.replace("\\", "\\\\")
        escaped = escaped.replace("*", "\\*")
        escaped = escaped.replace("_", "\\_")
        escaped = escaped.replace("`", "\\`")
        if in_table:
            escaped = escaped.replace("|", "\\|")
        return escaped

    def _strip_markdown_text(self, text: str) -> str:
        without_links = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        without_tags = re.sub(r"</?u>", "", without_links)
        without_formatting = without_tags.replace("***", "").replace("**", "").replace("*", "").replace("~~", "")
        without_list_marker = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", without_formatting)
        return without_list_marker.replace("\\", "").strip()

    @staticmethod
    def _safe_log_text(text: str) -> str:
        return text.encode("unicode_escape").decode("ascii")


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag
