from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple
from urllib.parse import urldefrag, urljoin, urlparse

from lxml import etree, html as lxml_html

from rag_lib.core.domain import Document
from rag_lib.html_processing import (
    extract_structural_blocks,
    render_blocks_as_markdown,
    strip_non_content_nodes,
)
from rag_lib.loaders.csv_excel import CSVLoader, ExcelLoader
from rag_lib.loaders.data_loaders import JsonLoader, TextLoader
from rag_lib.loaders.docx import DocXLoader
from rag_lib.loaders.html import HTMLLoader
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.pymupdf import PyMuPDFLoader

HTML_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}

_DOWNLOAD_EXTENSION_TO_KIND = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
    ".txt": "txt",
}

_DOWNLOAD_CONTENT_TYPE_TO_KIND = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/csv": "csv",
    "application/csv": "csv",
    "application/json": "json",
    "text/json": "json",
    "text/plain": "txt",
    "text/html": "html",
    "application/xhtml+xml": "html",
}

_KIND_TO_EXTENSION = {
    "pdf": ".pdf",
    "docx": ".docx",
    "html": ".html",
    "csv": ".csv",
    "xlsx": ".xlsx",
    "json": ".json",
    "txt": ".txt",
}


@dataclass(frozen=True)
class WebCleanupConfig:
    """
    Cleanup and link-discovery rules applied to parsed HTML before rendering.

    Attributes:
        ignored_classes: Elements matching these class rules are removed before rendering and link extraction.
        non_recursive_classes: Regular links originating from elements with these classes are not recursively crawled.
        navigation_classes: Elements matching these class rules are treated as navigation sources.
        navigation_styles: Elements with inline styles containing these normalized snippets are treated as navigation sources.
        navigation_texts: Exact normalized text markers for navigation controls (for example '<', '>', 'next').
        duplicate_tags: Tag names that participate in cross-page duplicate removal based on stable content signatures.
    """

    ignored_classes: Sequence[str] = ()
    non_recursive_classes: Sequence[str] = ()
    navigation_classes: Sequence[str] = ()
    navigation_styles: Sequence[str] = ()
    navigation_texts: Sequence[str] = ()
    duplicate_tags: Sequence[str] = ()


@dataclass(frozen=True)
class WebLink:
    """
    Normalized crawl candidate produced by HTML/custom/playwright extractors.

    Attributes:
        url: Absolute HTTP(S) URL.
        source_classes: Source element class tokens used for filters like non_recursive_classes.
        is_navigation: Marks same-depth traversal candidates.
        source_tag: Source element local tag name or extractor tag hint.
    """

    url: str
    source_classes: tuple[str, ...] = ()
    is_navigation: bool = False
    source_tag: str = ""


WebLinkInput = WebLink | str | tuple[str, Sequence[str]]
PDFRoutingHook = Callable[[str], tuple[list[Document], str]]


def normalize_url(url: str) -> str:
    """
    Normalize URL for crawl de-duplication.
    Removes fragment and trims whitespace.
    """
    cleaned = (url or "").strip()
    normalized, _ = urldefrag(cleaned)
    return normalized


def normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def get_header(headers: Mapping[str, str] | None, name: str) -> str:
    if not headers:
        return ""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return ""


def is_http_url(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme in {"http", "https"}


def url_extension(url: str) -> str:
    path = urlparse(url).path or ""
    return Path(path).suffix.lower()


def is_html_content_type(content_type: str | None) -> bool:
    return normalize_content_type(content_type) in HTML_CONTENT_TYPES


def is_html_response(url: str, content_type: str | None) -> bool:
    if is_html_content_type(content_type):
        return True
    return url_extension(url) in {".html", ".htm", ".xhtml"}


def parse_web_html_document(content: str | bytes) -> etree._Element:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    if isinstance(content, bytes):
        return lxml_html.fromstring(content, parser=parser)
    return lxml_html.fromstring(content.encode("utf-8"), parser=parser)


def resolve_absolute_links(html_content: str, base_url: str) -> list[str]:
    """
    Extract anchor links and normalize them into absolute HTTP(S) URLs.
    """
    root = parse_web_html_document(html_content)
    links: list[str] = []
    seen: set[str] = set()

    for href in root.xpath("//a[@href]/@href"):
        absolute = normalize_url(urljoin(base_url, href))
        if not absolute or not is_http_url(absolute) or absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)

    return links


def _split_class_tokens(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.split()
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split())
    ordered: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = part.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _class_rule_matches(element_classes: set[str], class_rule: str) -> bool:
    rule_tokens = [part.strip() for part in class_rule.split() if part.strip()]
    if not rule_tokens:
        return False
    if len(rule_tokens) == 1:
        return rule_tokens[0] in element_classes
    return all(token in element_classes for token in rule_tokens)


def _matches_any_class_rule(element: etree._Element, class_rules: Sequence[str]) -> bool:
    if not class_rules:
        return False
    element_classes = set(_split_class_tokens(element.get("class")))
    if not element_classes:
        return False
    return any(_class_rule_matches(element_classes, class_rule) for class_rule in class_rules)


def _iter_elements_matching_class_rules(
    document: etree._Element,
    class_rules: Sequence[str],
) -> list[etree._Element]:
    if not class_rules:
        return []
    matched: list[etree._Element] = []
    for element in document.xpath("//*"):
        if _matches_any_class_rule(element, class_rules):
            matched.append(element)
    return matched


def _normalize_style_value(style: str) -> str:
    return re.sub(r"\s+", "", style.lower())


def _iter_elements_matching_style_rules(
    document: etree._Element,
    style_rules: Sequence[str],
) -> list[etree._Element]:
    if not style_rules:
        return []

    normalized_rules = [
        _normalize_style_value(rule)
        for rule in style_rules
        if isinstance(rule, str) and rule.strip()
    ]
    if not normalized_rules:
        return []

    matched: list[etree._Element] = []
    for element in document.xpath("//*[@style]"):
        style_attr = element.get("style")
        if not style_attr:
            continue
        normalized_style = _normalize_style_value(style_attr)
        if any(rule in normalized_style for rule in normalized_rules):
            matched.append(element)
    return matched


def _normalize_text_value(text: str | None) -> str:
    return " ".join((text or "").split())


def _iter_navigation_text_elements(
    document: etree._Element,
    navigation_texts: Sequence[str],
) -> list[etree._Element]:
    if not navigation_texts:
        return []

    marker_set = {_normalize_text_value(item) for item in navigation_texts if isinstance(item, str) and item.strip()}
    if not marker_set:
        return []

    matched: list[etree._Element] = []
    for element in document.xpath("//a|//button"):
        text_value = _normalize_text_value("".join(element.itertext()))
        if text_value and text_value in marker_set:
            matched.append(element)
    return matched


def _element_local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    if "}" in element.tag:
        return element.tag.rsplit("}", 1)[-1].lower()
    return element.tag.lower()


def _safe_remove(element: etree._Element) -> None:
    parent = element.getparent()
    if parent is not None:
        parent.remove(element)


def _extract_anchor_links(
    root: etree._Element,
    *,
    base_url: str,
    is_navigation: bool,
) -> list[WebLink]:
    links: list[WebLink] = []
    seen: set[tuple[str, bool]] = set()
    for anchor in root.xpath(".//a[@href]"):
        href = anchor.get("href")
        if not href:
            continue
        url = normalize_url(urljoin(base_url, href))
        if not url or not is_http_url(url):
            continue
        key = (url, is_navigation)
        if key in seen:
            continue
        seen.add(key)
        links.append(
            WebLink(
                url=url,
                source_classes=_split_class_tokens(anchor.get("class")),
                is_navigation=is_navigation,
                source_tag=_element_local_name(anchor),
            )
        )
    return links


def _navigation_signature(element: etree._Element, base_url: str) -> str:
    hrefs: list[str] = []
    for href in element.xpath(".//a[@href]/@href"):
        normalized = normalize_url(urljoin(base_url, href))
        if normalized and is_http_url(normalized):
            hrefs.append(normalized)
    if hrefs:
        canonical = "\n".join(sorted(set(hrefs)))
    else:
        canonical = _normalize_text_value("".join(element.itertext()))
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def _duplicate_signature(element: etree._Element, base_url: str) -> str:
    text_parts: list[str] = []
    for part in element.itertext():
        normalized = " ".join(part.split())
        if normalized:
            text_parts.append(normalized)
    text = " ".join(text_parts)

    hrefs: list[str] = []
    for href in element.xpath(".//a[@href]/@href"):
        normalized = normalize_url(urljoin(base_url, href))
        if normalized and is_http_url(normalized):
            hrefs.append(normalized)

    tag_name = _element_local_name(element)
    payload = f"{tag_name}\n{text}\n{'|'.join(sorted(set(hrefs)))}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def normalize_web_link_input(item: WebLinkInput, *, base_url: str) -> WebLink | None:
    if isinstance(item, WebLink):
        url = normalize_url(urljoin(base_url, item.url))
        if not url or not is_http_url(url):
            return None
        return WebLink(
            url=url,
            source_classes=tuple(item.source_classes),
            is_navigation=item.is_navigation,
            source_tag=item.source_tag,
        )

    if isinstance(item, str):
        url = normalize_url(urljoin(base_url, item))
        if not url or not is_http_url(url):
            return None
        return WebLink(url=url)

    if isinstance(item, tuple) and len(item) == 2:
        url_part, class_part = item
        url = normalize_url(urljoin(base_url, str(url_part)))
        if not url or not is_http_url(url):
            return None
        return WebLink(url=url, source_classes=_split_class_tokens(class_part))

    raise TypeError("Unsupported link extractor item type.")


def merge_web_links(*groups: Sequence[WebLink]) -> list[WebLink]:
    merged: dict[tuple[str, bool], WebLink] = {}
    order: list[tuple[str, bool]] = []

    for group in groups:
        for link in group:
            key = (link.url, link.is_navigation)
            if key not in merged:
                merged[key] = link
                order.append(key)
                continue

            existing = merged[key]
            prefer_existing = len(existing.source_classes) > len(link.source_classes)
            if prefer_existing:
                continue
            if len(existing.source_classes) == len(link.source_classes) and existing.source_tag and not link.source_tag:
                continue
            merged[key] = link

    return [merged[key] for key in order]


def partition_web_links(links: Sequence[WebLink]) -> tuple[list[WebLink], list[WebLink]]:
    regular: list[WebLink] = []
    navigation: list[WebLink] = []
    for link in links:
        if link.is_navigation:
            navigation.append(link)
        else:
            regular.append(link)
    return regular, navigation


def cleanup_and_extract_web_links(
    document: etree._Element,
    *,
    base_url: str,
    cleanup_config: WebCleanupConfig | None,
    duplicate_hashes: set[str] | None = None,
    navigation_hashes: set[str] | None = None,
) -> tuple[list[WebLink], list[WebLink]]:
    config = cleanup_config or WebCleanupConfig()

    for element in _iter_elements_matching_class_rules(document, config.ignored_classes):
        _safe_remove(element)

    navigation_links: list[WebLink] = []
    class_navigation_elements = _iter_elements_matching_class_rules(document, config.navigation_classes)
    style_navigation_elements = _iter_elements_matching_style_rules(document, config.navigation_styles)
    text_navigation_elements = _iter_navigation_text_elements(document, config.navigation_texts)
    navigation_elements = [
        *class_navigation_elements,
        *style_navigation_elements,
        *text_navigation_elements,
    ]
    removable_navigation_ids = {
        *(id(element) for element in class_navigation_elements),
        *(id(element) for element in text_navigation_elements),
    }
    seen_navigation_elements: set[int] = set()
    for nav_element in navigation_elements:
        marker = id(nav_element)
        if marker in seen_navigation_elements:
            continue
        seen_navigation_elements.add(marker)

        signature = _navigation_signature(nav_element, base_url)
        already_processed = navigation_hashes is not None and signature in navigation_hashes
        if not already_processed:
            navigation_links.extend(
                _extract_anchor_links(nav_element, base_url=base_url, is_navigation=True)
            )
            if navigation_hashes is not None:
                navigation_hashes.add(signature)
        if id(nav_element) in removable_navigation_ids:
            _safe_remove(nav_element)

    if config.duplicate_tags and duplicate_hashes is not None:
        duplicate_set = {tag.lower().strip() for tag in config.duplicate_tags if tag and tag.strip()}
        if duplicate_set:
            to_remove: list[etree._Element] = []
            for element in document.xpath("//*"):
                if _element_local_name(element) not in duplicate_set:
                    continue
                signature = _duplicate_signature(element, base_url)
                if signature in duplicate_hashes:
                    to_remove.append(element)
                else:
                    duplicate_hashes.add(signature)
            for element in to_remove:
                _safe_remove(element)

    regular_links = _extract_anchor_links(document, base_url=base_url, is_navigation=False)
    regular_links = merge_web_links(regular_links)
    navigation_links = merge_web_links(navigation_links)
    return regular_links, navigation_links


def normalize_html_for_processing(content: str | bytes) -> bytes:
    """
    Normalize lenient web HTML into a stable HTML payload for web-mode processing.
    """
    root = parse_web_html_document(content)
    _prune_empty_lists(root)
    return lxml_html.tostring(root, encoding="utf-8", method="html")


def _has_meaningful_text(element: etree._Element) -> bool:
    text = "".join(part for part in element.itertext())
    normalized = text.replace("\xa0", " ").strip()
    return bool(normalized)


def _prune_empty_lists(root: etree._Element) -> None:
    """
    Web pages often contain structural list containers that have no text content.
    HTML strict conversion raises on these blocks, so drop them in web mode only.
    """
    for list_node in root.xpath("//ul|//ol"):
        items = list_node.xpath("./li")
        if not items:
            parent = list_node.getparent()
            if parent is not None:
                parent.remove(list_node)
            continue

        if any(_has_meaningful_text(item) for item in items):
            continue

        parent = list_node.getparent()
        if parent is not None:
            parent.remove(list_node)


def render_web_html_document(
    document: etree._Element,
    output_format: Literal["markdown", "html"],
) -> str:
    if output_format not in {"markdown", "html"}:
        raise ValueError("output_format must be 'markdown' or 'html'")

    _prune_empty_lists(document)
    strip_non_content_nodes(document)

    if output_format == "html":
        return lxml_html.tostring(document, encoding="unicode", method="html").strip()

    blocks = extract_structural_blocks(document)
    return render_blocks_as_markdown(blocks)


def render_web_html_content(
    content: str | bytes,
    output_format: Literal["markdown", "html"],
) -> str:
    document = parse_web_html_document(content)
    return render_web_html_document(document, output_format=output_format)


def _registrable_domain(hostname: str) -> str:
    if not hostname:
        return ""
    host = hostname.lower().strip(".")
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    return ".".join(parts[-2:])


def _host_from_value(value: str) -> str:
    parsed = urlparse(value)
    if parsed.netloc:
        return parsed.hostname or ""
    return value.split(":", 1)[0]


def is_url_in_scope(
    url: str,
    start_url: str,
    crawl_scope: Literal["same_host", "same_domain", "allowed_domains", "allow_all"],
    allowed_domains: Optional[Sequence[str]] = None,
) -> bool:
    candidate_host = (urlparse(url).hostname or "").lower()
    start_host = (urlparse(start_url).hostname or "").lower()

    if crawl_scope == "allow_all":
        return True

    if not candidate_host:
        return False

    if crawl_scope == "same_host":
        return candidate_host == start_host

    if crawl_scope == "same_domain":
        return _registrable_domain(candidate_host) == _registrable_domain(start_host)

    if crawl_scope == "allowed_domains":
        for item in allowed_domains or []:
            allowed_host = _host_from_value(item).lower().strip()
            if not allowed_host:
                continue
            if candidate_host == allowed_host or candidate_host.endswith(f".{allowed_host}"):
                return True
        return False

    raise ValueError(f"Unknown crawl_scope: {crawl_scope}")


def _filename_from_content_disposition(content_disposition: str) -> str:
    # RFC 6266 filename*=UTF-8''foo.ext
    star_match = re.search(r"filename\*\s*=\s*(?:UTF-8''|)([^;]+)", content_disposition, flags=re.I)
    if star_match:
        return star_match.group(1).strip().strip("\"'")
    simple_match = re.search(r"filename\s*=\s*\"?([^\";]+)\"?", content_disposition, flags=re.I)
    if simple_match:
        return simple_match.group(1).strip()
    return ""


def resolve_download_filename(url: str, headers: Mapping[str, str] | None, kind: str | None) -> str:
    content_disposition = get_header(headers, "content-disposition")
    filename = _filename_from_content_disposition(content_disposition)
    if not filename:
        filename = Path(urlparse(url).path).name
    if not filename:
        filename = "download"
    suffix = Path(filename).suffix
    if not suffix and kind in _KIND_TO_EXTENSION:
        filename = f"{filename}{_KIND_TO_EXTENSION[kind]}"
    return filename


def infer_download_kind(
    url: str,
    content_type: str | None,
    headers: Mapping[str, str] | None,
) -> str | None:
    normalized_type = normalize_content_type(content_type)
    extension = url_extension(url)
    if extension in _DOWNLOAD_EXTENSION_TO_KIND:
        return _DOWNLOAD_EXTENSION_TO_KIND[extension]

    if normalized_type in _DOWNLOAD_CONTENT_TYPE_TO_KIND:
        return _DOWNLOAD_CONTENT_TYPE_TO_KIND[normalized_type]

    content_disposition = get_header(headers, "content-disposition")
    if "attachment" in content_disposition.lower():
        filename = _filename_from_content_disposition(content_disposition)
        ext = Path(filename).suffix.lower()
        if ext in _DOWNLOAD_EXTENSION_TO_KIND:
            return _DOWNLOAD_EXTENSION_TO_KIND[ext]

    return None


def is_download_response(
    url: str,
    content_type: str | None,
    headers: Mapping[str, str] | None,
) -> bool:
    if infer_download_kind(url=url, content_type=content_type, headers=headers):
        return True
    content_disposition = get_header(headers, "content-disposition")
    return "attachment" in content_disposition.lower()


def route_download_content_to_documents(
    *,
    content_bytes: bytes,
    source_url: str,
    content_type: str | None,
    headers: Mapping[str, str] | None,
    output_format: Literal["markdown", "html"],
    pdf_routing_hook: PDFRoutingHook | None = None,
) -> list[Document]:
    """
    Persist downloaded bytes temporarily and route to the matching file loader.
    """
    kind = infer_download_kind(url=source_url, content_type=content_type, headers=headers)
    if kind is None:
        raise ValueError(f"Unsupported downloadable content type for URL: {source_url}")

    filename = resolve_download_filename(source_url, headers, kind)
    suffix = _KIND_TO_EXTENSION.get(kind, Path(filename).suffix or ".bin")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content_bytes)
            temp_path = temp_file.name

        if kind == "pdf":
            if pdf_routing_hook is not None:
                docs, routed_loader = pdf_routing_hook(temp_path)
            else:
                try:
                    docs = PDFLoader(temp_path, parse_mode="text").load()
                    routed_loader = "PDFLoader"
                except Exception:
                    docs = PyMuPDFLoader(temp_path, output_format="markdown").load()
                    routed_loader = "PyMuPDFLoader"
        elif kind == "docx":
            docs = DocXLoader(temp_path).load()
            routed_loader = "DocXLoader"
        elif kind == "html":
            docs = HTMLLoader(temp_path, output_format=output_format).load()
            routed_loader = "HTMLLoader"
        elif kind == "csv":
            docs = CSVLoader(temp_path, output_format="markdown").load()
            routed_loader = "CSVLoader"
        elif kind == "xlsx":
            docs = ExcelLoader(temp_path, output_format="markdown").load()
            routed_loader = "ExcelLoader"
        elif kind == "json":
            docs = JsonLoader(temp_path, output_format="json").load()
            routed_loader = "JsonLoader"
        elif kind == "txt":
            docs = TextLoader(temp_path).load()
            routed_loader = "TextLoader"
        else:
            raise ValueError(f"Unsupported download kind: {kind}")

        wrapped: list[Document] = []
        normalized_content_type = normalize_content_type(content_type)
        for doc in docs:
            metadata = dict(doc.metadata or {})
            if "source" in metadata:
                metadata["download_source"] = metadata["source"]
            metadata["source"] = source_url
            metadata["source_type"] = "web_download"
            metadata["download_content_type"] = normalized_content_type
            metadata["download_filename"] = filename
            metadata["routed_loader"] = routed_loader
            wrapped.append(Document(page_content=doc.page_content, metadata=metadata))

        return wrapped
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
