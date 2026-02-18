from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NS = {
    "w": W_NS,
    "r": R_NS,
    "a": A_NS,
}


@dataclass
class _Relationship:
    target: str
    rel_type: str
    target_mode: Optional[str] = None


@dataclass
class _StyleDefinition:
    style_id: str
    style_type: str
    name: str
    based_on: Optional[str]
    outline_level: Optional[int]
    num_id: Optional[str]
    ilvl: Optional[int]
    bold: Optional[bool]
    italic: Optional[bool]
    underline: Optional[bool]
    strike: Optional[bool]


@dataclass
class _ResolvedStyle:
    style_id: str = ""
    style_type: str = ""
    name: str = ""
    based_on: Optional[str] = None
    outline_level: Optional[int] = None
    num_id: Optional[str] = None
    ilvl: Optional[int] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    strike: Optional[bool] = None


@dataclass
class _ListLevelDefinition:
    num_fmt: str = "bullet"
    lvl_text: str = ""
    start: int = 1


@dataclass
class _InlineStats:
    total_chars: int = 0
    bold_chars: int = 0

    def add(self, other: "_InlineStats") -> None:
        self.total_chars += other.total_chars
        self.bold_chars += other.bold_chars


@dataclass
class _Block:
    text: str
    kind: str
    list_key: Optional[str] = None


@dataclass
class _ParagraphProperties:
    style_id: Optional[str]
    outline_level: Optional[int]
    num_id: Optional[str]
    ilvl: Optional[int]


class _StyleResolver:
    def __init__(self, styles_root: Optional[ET.Element]):
        self._styles_root = styles_root
        self._paragraph_defs: Dict[str, _StyleDefinition] = {}
        self._character_defs: Dict[str, _StyleDefinition] = {}
        self._paragraph_cache: Dict[str, _ResolvedStyle] = {}
        self._character_cache: Dict[str, _ResolvedStyle] = {}
        if styles_root is not None:
            self._parse_styles(styles_root)

    def resolve_paragraph(self, style_id: Optional[str]) -> _ResolvedStyle:
        if not style_id:
            return _ResolvedStyle()
        if style_id in self._paragraph_cache:
            return self._paragraph_cache[style_id]
        resolved = self._resolve_recursive(style_id, self._paragraph_defs, self._paragraph_cache)
        self._paragraph_cache[style_id] = resolved
        return resolved

    def resolve_character(self, style_id: Optional[str]) -> _ResolvedStyle:
        if not style_id:
            return _ResolvedStyle()
        if style_id in self._character_cache:
            return self._character_cache[style_id]
        resolved = self._resolve_recursive(style_id, self._character_defs, self._character_cache)
        self._character_cache[style_id] = resolved
        return resolved

    def _resolve_recursive(
        self,
        style_id: str,
        defs: Dict[str, _StyleDefinition],
        cache: Dict[str, _ResolvedStyle],
        seen: Optional[set[str]] = None,
    ) -> _ResolvedStyle:
        if style_id in cache:
            return cache[style_id]

        style_def = defs.get(style_id)
        if style_def is None:
            return _ResolvedStyle(style_id=style_id)

        if seen is None:
            seen = set()
        if style_id in seen:
            return _ResolvedStyle(style_id=style_id)
        seen.add(style_id)

        parent = _ResolvedStyle()
        if style_def.based_on:
            parent = self._resolve_recursive(style_def.based_on, defs, cache, seen)

        merged = _ResolvedStyle(
            style_id=style_def.style_id,
            style_type=style_def.style_type or parent.style_type,
            name=style_def.name or parent.name,
            based_on=style_def.based_on,
            outline_level=self._coalesce(style_def.outline_level, parent.outline_level),
            num_id=self._coalesce(style_def.num_id, parent.num_id),
            ilvl=self._coalesce(style_def.ilvl, parent.ilvl),
            bold=self._coalesce(style_def.bold, parent.bold),
            italic=self._coalesce(style_def.italic, parent.italic),
            underline=self._coalesce(style_def.underline, parent.underline),
            strike=self._coalesce(style_def.strike, parent.strike),
        )

        cache[style_id] = merged
        return merged

    def _parse_styles(self, styles_root: ET.Element) -> None:
        for style in styles_root.findall("w:style", NS):
            style_type = style.get(f"{{{W_NS}}}type", "")
            style_id = style.get(f"{{{W_NS}}}styleId", "")
            if not style_id:
                continue

            name = ""
            name_elem = style.find("w:name", NS)
            if name_elem is not None:
                name = name_elem.get(f"{{{W_NS}}}val", "")

            based_on = None
            based_on_elem = style.find("w:basedOn", NS)
            if based_on_elem is not None:
                based_on = based_on_elem.get(f"{{{W_NS}}}val")

            ppr = style.find("w:pPr", NS)
            rpr = style.find("w:rPr", NS)

            outline_level = None
            num_id = None
            ilvl = None

            if ppr is not None:
                outline_elem = ppr.find("w:outlineLvl", NS)
                if outline_elem is not None:
                    outline_level = self._safe_int(outline_elem.get(f"{{{W_NS}}}val"))

                numpr = ppr.find("w:numPr", NS)
                if numpr is not None:
                    num_id, ilvl = self._parse_numpr(numpr)

            bold = italic = underline = strike = None
            if rpr is not None:
                bold = _read_on_off(rpr.find("w:b", NS))
                italic = _read_on_off(rpr.find("w:i", NS))
                underline = _read_on_off(rpr.find("w:u", NS))
                strike = _read_on_off(rpr.find("w:strike", NS))

            style_def = _StyleDefinition(
                style_id=style_id,
                style_type=style_type,
                name=name,
                based_on=based_on,
                outline_level=outline_level,
                num_id=num_id,
                ilvl=ilvl,
                bold=bold,
                italic=italic,
                underline=underline,
                strike=strike,
            )

            if style_type == "character":
                self._character_defs[style_id] = style_def
            else:
                self._paragraph_defs[style_id] = style_def

    @staticmethod
    def _coalesce(primary: Optional[object], fallback: Optional[object]) -> Optional[object]:
        return primary if primary is not None else fallback

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_numpr(numpr: ET.Element) -> Tuple[Optional[str], Optional[int]]:
        num_id = None
        ilvl = None

        num_id_elem = numpr.find("w:numId", NS)
        if num_id_elem is not None:
            num_id = num_id_elem.get(f"{{{W_NS}}}val")

        ilvl_elem = numpr.find("w:ilvl", NS)
        if ilvl_elem is not None:
            try:
                ilvl = int(ilvl_elem.get(f"{{{W_NS}}}val", "0"))
            except (TypeError, ValueError):
                ilvl = 0

        return num_id, ilvl


class _NumberingResolver:
    def __init__(self, numbering_root: Optional[ET.Element]):
        self._num_to_abstract: Dict[str, str] = {}
        self._abstract_levels: Dict[str, Dict[int, _ListLevelDefinition]] = {}
        self._overrides: Dict[Tuple[str, int], _ListLevelDefinition] = {}
        if numbering_root is not None:
            self._parse(numbering_root)

    def resolve(self, num_id: str, ilvl: int) -> Optional[_ListLevelDefinition]:
        override = self._overrides.get((num_id, ilvl))
        if override is not None:
            return override

        abstract_id = self._num_to_abstract.get(num_id)
        if not abstract_id:
            return None

        levels = self._abstract_levels.get(abstract_id, {})
        level = levels.get(ilvl)
        if level is not None:
            return level

        return levels.get(0)

    def _parse(self, numbering_root: ET.Element) -> None:
        for abstract in numbering_root.findall("w:abstractNum", NS):
            abstract_id = abstract.get(f"{{{W_NS}}}abstractNumId")
            if not abstract_id:
                continue

            levels: Dict[int, _ListLevelDefinition] = {}
            for lvl in abstract.findall("w:lvl", NS):
                ilvl_raw = lvl.get(f"{{{W_NS}}}ilvl", "0")
                try:
                    ilvl = int(ilvl_raw)
                except (TypeError, ValueError):
                    ilvl = 0

                levels[ilvl] = self._parse_level_definition(lvl)

            self._abstract_levels[abstract_id] = levels

        for num in numbering_root.findall("w:num", NS):
            num_id = num.get(f"{{{W_NS}}}numId")
            if not num_id:
                continue

            abstract_id_elem = num.find("w:abstractNumId", NS)
            if abstract_id_elem is not None:
                abstract_id = abstract_id_elem.get(f"{{{W_NS}}}val")
                if abstract_id:
                    self._num_to_abstract[num_id] = abstract_id

            for lvl_override in num.findall("w:lvlOverride", NS):
                ilvl_raw = lvl_override.get(f"{{{W_NS}}}ilvl", "0")
                try:
                    ilvl = int(ilvl_raw)
                except (TypeError, ValueError):
                    ilvl = 0

                lvl = lvl_override.find("w:lvl", NS)
                if lvl is not None:
                    self._overrides[(num_id, ilvl)] = self._parse_level_definition(lvl)
                    continue

                start_override = lvl_override.find("w:startOverride", NS)
                if start_override is not None:
                    start_raw = start_override.get(f"{{{W_NS}}}val", "1")
                    try:
                        start_value = int(start_raw)
                    except (TypeError, ValueError):
                        start_value = 1
                    base = self.resolve(num_id, ilvl) or _ListLevelDefinition()
                    self._overrides[(num_id, ilvl)] = _ListLevelDefinition(
                        num_fmt=base.num_fmt,
                        lvl_text=base.lvl_text,
                        start=start_value,
                    )

    @staticmethod
    def _parse_level_definition(level_elem: ET.Element) -> _ListLevelDefinition:
        num_fmt = "bullet"
        lvl_text = ""
        start = 1

        num_fmt_elem = level_elem.find("w:numFmt", NS)
        if num_fmt_elem is not None:
            num_fmt = num_fmt_elem.get(f"{{{W_NS}}}val", "bullet")

        lvl_text_elem = level_elem.find("w:lvlText", NS)
        if lvl_text_elem is not None:
            lvl_text = lvl_text_elem.get(f"{{{W_NS}}}val", "")

        start_elem = level_elem.find("w:start", NS)
        if start_elem is not None:
            try:
                start = int(start_elem.get(f"{{{W_NS}}}val", "1"))
            except (TypeError, ValueError):
                start = 1

        return _ListLevelDefinition(num_fmt=num_fmt, lvl_text=lvl_text, start=start)


@dataclass
class _ConversionContext:
    style_resolver: _StyleResolver
    numbering_resolver: _NumberingResolver
    relationships: Dict[str, _Relationship]
    list_counters: Dict[Tuple[str, int], int] = field(default_factory=dict)


class DocXLoader:
    """
    DOCX -> Markdown loader preserving structural and inline semantics.

    Output:
    - Returns one Document with markdown in page_content.
    - Returns [] on failures (tolerant behavior).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with zipfile.ZipFile(self.file_path, "r") as docx_file:
                document_root = self._read_xml_part(docx_file, "word/document.xml", required=True)
                if document_root is None:
                    return []

                styles_root = self._read_xml_part(docx_file, "word/styles.xml", required=False)
                numbering_root = self._read_xml_part(docx_file, "word/numbering.xml", required=False)
                relationships = self._read_relationships(docx_file)

            style_resolver = _StyleResolver(styles_root)
            numbering_resolver = _NumberingResolver(numbering_root)
            context = _ConversionContext(
                style_resolver=style_resolver,
                numbering_resolver=numbering_resolver,
                relationships=relationships,
            )

            markdown = self._convert_document(document_root, context)
            if not markdown.strip():
                return []

            return [
                Document(
                    page_content=markdown,
                    metadata={
                        "source": self.file_path,
                        "source_type": "docx",
                        "output_format": "markdown",
                    },
                )
            ]

        except Exception as exc:
            logger.error(f"Failed to load DOCX '{self.file_path}': {exc}")
            return []

    def _read_xml_part(
        self,
        docx_file: zipfile.ZipFile,
        part_name: str,
        required: bool,
    ) -> Optional[ET.Element]:
        try:
            raw = docx_file.read(part_name)
        except KeyError:
            if required:
                logger.error(f"DOCX part is missing: {part_name}")
            return None
        except Exception as exc:
            if required:
                logger.error(f"Failed to read DOCX part '{part_name}': {exc}")
            return None

        try:
            return ET.fromstring(raw)
        except ET.ParseError as exc:
            if required:
                logger.error(f"Invalid XML in DOCX part '{part_name}': {exc}")
            return None

    def _read_relationships(self, docx_file: zipfile.ZipFile) -> Dict[str, _Relationship]:
        rel_root = self._read_xml_part(docx_file, "word/_rels/document.xml.rels", required=False)
        if rel_root is None:
            return {}

        relationships: Dict[str, _Relationship] = {}
        rel_tag = f"{{{PKG_REL_NS}}}Relationship"

        for rel in rel_root.findall(rel_tag):
            rel_id = rel.get("Id")
            target = rel.get("Target")
            rel_type = rel.get("Type", "")
            target_mode = rel.get("TargetMode")

            if not rel_id or not target:
                continue

            normalized_target = self._normalize_relationship_target(target, target_mode)
            relationships[rel_id] = _Relationship(
                target=normalized_target,
                rel_type=rel_type,
                target_mode=target_mode,
            )

        return relationships

    def _normalize_relationship_target(self, target: str, target_mode: Optional[str]) -> str:
        if target_mode == "External":
            return target
        if target.startswith("#"):
            return target
        if target.startswith("word/"):
            return target
        return str((PurePosixPath("word") / target).as_posix())

    def _convert_document(self, document_root: ET.Element, context: _ConversionContext) -> str:
        body = document_root.find("w:body", NS)
        if body is None:
            return ""

        blocks: List[_Block] = []
        for child in body:
            local = _local_name(child.tag)
            if local == "p":
                block = self._convert_paragraph(child, context)
                if block is not None:
                    blocks.append(block)
            elif local == "tbl":
                table_md = self._convert_table(child, context)
                if table_md.strip():
                    blocks.append(_Block(text=table_md, kind="table"))

        return self._assemble_blocks(blocks)

    def _convert_paragraph(self, paragraph: ET.Element, context: _ConversionContext) -> Optional[_Block]:
        props = self._read_paragraph_properties(paragraph)
        para_style = context.style_resolver.resolve_paragraph(props.style_id)

        num_id = props.num_id or para_style.num_id
        ilvl = props.ilvl if props.ilvl is not None else para_style.ilvl
        ilvl = ilvl if ilvl is not None else 0

        text, stats = self._convert_inline_children(
            paragraph,
            context,
            para_style=para_style,
            in_table=False,
        )

        if not text.strip():
            return None

        heading_level = self._resolve_heading_level(props, para_style)
        heading_text = self._strip_inline_markup(text)

        if num_id:
            item = self._format_list_item(text.strip(), num_id, ilvl, context)
            return _Block(text=item, kind="list", list_key=num_id)

        if heading_level is None:
            heading_level = self._infer_heading_level(heading_text, stats)

        if heading_level is not None:
            heading_level = max(1, min(6, heading_level))
            return _Block(text=f"{'#' * heading_level} {heading_text.strip()}", kind="heading")

        return _Block(text=text.strip(), kind="paragraph")

    def _resolve_heading_level(
        self,
        props: _ParagraphProperties,
        style: _ResolvedStyle,
    ) -> Optional[int]:
        if props.outline_level is not None:
            return props.outline_level + 1

        if style.outline_level is not None:
            return style.outline_level + 1

        style_name = (style.name or "").strip().lower()
        if "heading" in style_name:
            match = re.search(r"heading\s*(\d+)", style_name)
            if match:
                return int(match.group(1))

        style_id = (style.style_id or "").strip()
        match = re.search(r"heading\s*(\d+)", style_id, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        return None

    def _infer_heading_level(self, text: str, stats: _InlineStats) -> Optional[int]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return None

        if len(normalized) < 4 or len(normalized) > 120:
            return None

        if stats.total_chars <= 0:
            return None

        bold_ratio = stats.bold_chars / max(1, stats.total_chars)
        if bold_ratio < 0.75:
            return None

        numbered_match = re.match(r"^(\d+(?:\.\d+)*)(?:[\.)])?\s+\S+", normalized)
        if numbered_match:
            depth = numbered_match.group(1).count(".") + 1
            return min(6, max(2, depth + 1))

        roman_match = re.match(r"^([IVXLCDM]+)[\.)]\s+\S+", normalized)
        if roman_match:
            return 2

        if normalized.endswith((".", "!", "?")) and not normalized.endswith(":"):
            return None

        if re.match(r"^[A-Z\u0410-\u042F\u0401][^.!?]{2,120}$", normalized):
            if len(normalized.split()) <= 12:
                return 2

        return None

    def _read_paragraph_properties(self, paragraph: ET.Element) -> _ParagraphProperties:
        ppr = paragraph.find("w:pPr", NS)
        if ppr is None:
            return _ParagraphProperties(style_id=None, outline_level=None, num_id=None, ilvl=None)

        style_id = None
        pstyle = ppr.find("w:pStyle", NS)
        if pstyle is not None:
            style_id = pstyle.get(f"{{{W_NS}}}val")

        outline_level = None
        outline = ppr.find("w:outlineLvl", NS)
        if outline is not None:
            try:
                outline_level = int(outline.get(f"{{{W_NS}}}val", "0"))
            except (TypeError, ValueError):
                outline_level = 0

        num_id = None
        ilvl = None
        numpr = ppr.find("w:numPr", NS)
        if numpr is not None:
            num_id_elem = numpr.find("w:numId", NS)
            ilvl_elem = numpr.find("w:ilvl", NS)
            if num_id_elem is not None:
                num_id = num_id_elem.get(f"{{{W_NS}}}val")
            if ilvl_elem is not None:
                try:
                    ilvl = int(ilvl_elem.get(f"{{{W_NS}}}val", "0"))
                except (TypeError, ValueError):
                    ilvl = 0

        return _ParagraphProperties(
            style_id=style_id,
            outline_level=outline_level,
            num_id=num_id,
            ilvl=ilvl,
        )

    def _convert_inline_children(
        self,
        parent: ET.Element,
        context: _ConversionContext,
        para_style: _ResolvedStyle,
        in_table: bool,
    ) -> Tuple[str, _InlineStats]:
        parts: List[str] = []
        stats = _InlineStats()

        for child in list(parent):
            local = _local_name(child.tag)
            if local == "r":
                run_text, run_stats = self._convert_run(child, context, para_style, in_table=in_table)
                if run_text:
                    parts.append(run_text)
                stats.add(run_stats)
            elif local == "hyperlink":
                link_text, link_stats = self._convert_hyperlink(child, context, para_style, in_table=in_table)
                if link_text:
                    parts.append(link_text)
                stats.add(link_stats)
            else:
                nested_text, nested_stats = self._convert_inline_children(
                    child,
                    context,
                    para_style=para_style,
                    in_table=in_table,
                )
                if nested_text:
                    parts.append(nested_text)
                stats.add(nested_stats)

        return "".join(parts), stats

    def _convert_run(
        self,
        run: ET.Element,
        context: _ConversionContext,
        para_style: _ResolvedStyle,
        in_table: bool,
    ) -> Tuple[str, _InlineStats]:
        rpr = run.find("w:rPr", NS)

        run_style = _ResolvedStyle()
        if rpr is not None:
            run_style_elem = rpr.find("w:rStyle", NS)
            if run_style_elem is not None:
                run_style_id = run_style_elem.get(f"{{{W_NS}}}val")
                run_style = context.style_resolver.resolve_character(run_style_id)

        bold = self._resolve_effective_flag(rpr, "b", run_style.bold, para_style.bold)
        italic = self._resolve_effective_flag(rpr, "i", run_style.italic, para_style.italic)
        underline = self._resolve_effective_flag(rpr, "u", run_style.underline, para_style.underline)
        strike = self._resolve_effective_flag(rpr, "strike", run_style.strike, para_style.strike)

        segments: List[str] = []

        for node in list(run):
            local = _local_name(node.tag)
            if local == "t":
                segments.append(node.text or "")
            elif local == "tab":
                segments.append("\t")
            elif local in {"br", "cr"}:
                segments.append("\n")
            elif local == "noBreakHyphen":
                segments.append("-")
            elif local == "drawing":
                placeholder = self._extract_drawing_placeholder(node, context.relationships)
                if placeholder:
                    segments.append(placeholder)

        raw_text = self._normalize_text("".join(segments))
        if not raw_text:
            return "", _InlineStats()

        stats = _InlineStats(
            total_chars=len(raw_text.strip()),
            bold_chars=len(raw_text.strip()) if bold else 0,
        )

        escaped = self._escape_markdown(raw_text, in_table=in_table)
        formatted = self._apply_formatting(escaped, bold=bold, italic=italic, underline=underline, strike=strike)
        return formatted, stats

    def _convert_hyperlink(
        self,
        hyperlink: ET.Element,
        context: _ConversionContext,
        para_style: _ResolvedStyle,
        in_table: bool,
    ) -> Tuple[str, _InlineStats]:
        text, stats = self._convert_inline_children(
            hyperlink,
            context,
            para_style=para_style,
            in_table=in_table,
        )
        if not text.strip():
            return "", stats

        rid = hyperlink.get(f"{{{R_NS}}}id")
        anchor = hyperlink.get(f"{{{W_NS}}}anchor")

        target = None
        if rid:
            rel = context.relationships.get(rid)
            if rel is not None:
                target = rel.target

        if target is None and anchor:
            target = f"#{anchor}"

        if target is None:
            return text, stats

        safe_target = target.replace(" ", "%20")
        return f"[{text}]({safe_target})", stats

    def _extract_drawing_placeholder(
        self,
        drawing: ET.Element,
        relationships: Dict[str, _Relationship],
    ) -> str:
        blip = drawing.find(".//a:blip", NS)
        if blip is None:
            return ""

        rid = blip.get(f"{{{R_NS}}}embed")
        if not rid:
            return ""

        rel = relationships.get(rid)
        if rel is not None:
            target = rel.target
            name = PurePosixPath(target).name or target
            return f"[image: {name}]"

        return f"[image: {rid}]"

    def _format_list_item(
        self,
        text: str,
        num_id: str,
        ilvl: int,
        context: _ConversionContext,
    ) -> str:
        level_def = context.numbering_resolver.resolve(num_id, ilvl) or _ListLevelDefinition()

        for key in list(context.list_counters.keys()):
            if key[0] == num_id and key[1] > ilvl:
                del context.list_counters[key]

        counter_key = (num_id, ilvl)
        if counter_key not in context.list_counters:
            context.list_counters[counter_key] = max(1, level_def.start)
        else:
            context.list_counters[counter_key] += 1

        counter = context.list_counters[counter_key]
        marker = self._list_marker(level_def.num_fmt, counter)
        indent = "  " * max(0, ilvl)
        return f"{indent}{marker} {text}"

    def _list_marker(self, num_fmt: str, counter: int) -> str:
        fmt = (num_fmt or "").lower()
        if fmt == "bullet":
            return "-"

        if fmt in {"lowerroman", "upperroman"}:
            roman = _to_roman(counter)
            return f"{roman.lower() if fmt == 'lowerroman' else roman}."

        if fmt in {"lowerletter", "upperletter"}:
            alpha = _to_alpha(counter)
            return f"{alpha.lower() if fmt == 'lowerletter' else alpha.upper()}."

        return f"{counter}."

    def _convert_table(self, table: ET.Element, context: _ConversionContext) -> str:
        rows: List[List[str]] = []

        for tr in table.findall("w:tr", NS):
            cells: List[str] = []
            for tc in tr.findall("w:tc", NS):
                cells.append(self._convert_table_cell(tc, context))
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

        header = normalized_rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]

        for row in normalized_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _convert_table_cell(self, cell: ET.Element, context: _ConversionContext) -> str:
        parts: List[str] = []

        for paragraph in cell.findall("w:p", NS):
            props = self._read_paragraph_properties(paragraph)
            para_style = context.style_resolver.resolve_paragraph(props.style_id)
            text, _ = self._convert_inline_children(
                paragraph,
                context,
                para_style=para_style,
                in_table=True,
            )
            clean = text.strip()
            if clean:
                parts.append(clean)

        if not parts:
            return ""

        combined = "<br>".join(parts)
        return combined.replace("|", "\\|")

    def _assemble_blocks(self, blocks: List[_Block]) -> str:
        if not blocks:
            return ""

        lines: List[str] = []
        previous: Optional[_Block] = None

        for block in blocks:
            if previous is not None:
                keep_list_flow = (
                    previous.kind == "list"
                    and block.kind == "list"
                    and previous.list_key == block.list_key
                )
                if not keep_list_flow:
                    lines.append("")

            lines.append(block.text)
            previous = block

        return "\n".join(lines).strip()

    def _resolve_effective_flag(
        self,
        rpr: Optional[ET.Element],
        tag: str,
        run_style_default: Optional[bool],
        para_style_default: Optional[bool],
    ) -> bool:
        explicit = None
        if rpr is not None:
            explicit = _read_on_off(rpr.find(f"w:{tag}", NS))

        if explicit is not None:
            return explicit
        if run_style_default is not None:
            return run_style_default
        if para_style_default is not None:
            return para_style_default
        return False

    def _apply_formatting(
        self,
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

    def _escape_markdown(self, text: str, in_table: bool) -> str:
        escaped = text
        escaped = escaped.replace("\\", "\\\\")
        escaped = escaped.replace("*", "\\*")
        escaped = escaped.replace("_", "\\_")
        escaped = escaped.replace("`", "\\`")
        if in_table:
            escaped = escaped.replace("|", "\\|")
        return escaped

    def _normalize_text(self, text: str) -> str:
        return (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\xa0", " ")
            .replace("\u202f", " ")
        )

    def _strip_inline_markup(self, text: str) -> str:
        unlinked = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        ununderlined = re.sub(r"</?u>", "", unlinked)
        unformatted = ununderlined.replace("***", "").replace("**", "").replace("*", "").replace("~~", "")
        return unformatted.replace("\\", "")


def _read_on_off(elem: Optional[ET.Element]) -> Optional[bool]:
    if elem is None:
        return None

    value = elem.get(f"{{{W_NS}}}val")
    if value is None:
        return True

    value_lc = value.strip().lower()
    if value_lc in {"0", "false", "off", "no"}:
        return False
    if value_lc in {"1", "true", "on", "yes"}:
        return True

    return True


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _to_roman(value: int) -> str:
    if value <= 0:
        return str(value)

    numerals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]

    result: List[str] = []
    remaining = value
    for number, numeral in numerals:
        while remaining >= number:
            result.append(numeral)
            remaining -= number
    return "".join(result)


def _to_alpha(value: int) -> str:
    if value <= 0:
        return str(value)

    chars: List[str] = []
    current = value
    while current > 0:
        current -= 1
        chars.append(chr(ord("A") + (current % 26)))
        current //= 26
    return "".join(reversed(chars))
