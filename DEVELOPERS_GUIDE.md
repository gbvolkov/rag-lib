# rag-lib Developer's Guide

This guide is aligned to the current code in `src/rag_lib`.

## 1. Version and Runtime

- Package name: `rag-lib`
- Current project version (`pyproject.toml`): `0.2.1`
- Supported Python: `>=3.12,<3.14`

To check installed package version:

```bash
python -c "import importlib.metadata as m; print(m.version('rag-lib'))"
```

## 2. Installation

### 2.1 Base install

```bash
git clone https://github.com/your-org/rag-lib.git
cd rag-lib
pip install -e .
```

### 2.2 Optional extras

- `rag-lib[miner_u]`: MinerU / Magic-PDF integration
- `rag-lib[graph]`: Neo4j graph backend support
- `rag-lib[raptor]`: RAPTOR clustering dependencies
- `rag-lib[pymupdf]`: PyMuPDF markdown/html loader support
- `rag-lib[web]`: Playwright browser backend for WebLoader/AsyncWebLoader

Example:

```bash
pip install -e ".[miner_u,graph,raptor,pymupdf]"
```

## 3. Core Domain Models and Enums

### 3.1 `SegmentType` enum (`rag_lib.core.domain`)

```python
SegmentType = {
    "text",
    "table",
    "image",
    "audio",
    "code",
    "other",
}
```

### 3.2 `Segment` model (`rag_lib.core.domain.Segment`)

Fields:

- `content: str`
- `metadata: Dict[str, Any] = {}`
- `segment_id: Optional[str] = None`
- `parent_id: Optional[str] = None`
- `level: int = 0`
- `path: List[str] = []`
- `type: SegmentType = SegmentType.TEXT`
- `original_format: str = "text"`

Method:

- `to_langchain() -> langchain_core.documents.Document`

### 3.3 Top-level exports (`rag_lib.__init__`)

`rag_lib` currently exports:

- `Segment`
- `Indexer`
- `PDFLoader`
- `PyMuPDFLoader`
- `DocXLoader`
- `HTMLLoader`
- `WebLoader`
- `AsyncWebLoader`
- `WebCleanupConfig`
- `WebLink`
- `PlaywrightExtractionConfig`
- `PlaywrightNavigationConfig`
- `PlaywrightProfileConfig`
- `build_sync_playwright_extractor`
- `build_async_playwright_extractor`
- `get_playwright_profile_defaults`
- `SemanticChunker`
- `HTMLSplitter`
- `SegmentEnricher`

## 4. Loaders (`rag_lib.loaders`)

All loaders return `List[langchain_core.documents.Document]`.

### 4.1 Signatures and return behavior

#### `PDFLoader` (`rag_lib.loaders.pdf`)

```python
PDFLoader(
    file_path: str,
    parse_mode: str = "text",          # "text" | "table"
    summarizer: Optional[TableSummarizer] = None,
    backend: Optional[str] = None,
)
```

- `load()` in `text` mode returns one merged text `Document`.
- `load()` in `table` mode returns one `Document` per extracted table.
- Raises for invalid `parse_mode` or backend/dependency failures.

#### `PyMuPDFLoader` (`rag_lib.loaders.pymupdf`)

```python
PyMuPDFLoader(file_path: str, output_format: str = "markdown")  # "markdown" | "html"
```

- Returns one `Document`.
- Metadata includes `parser`, `output_format`, optional `page_count`.
- Raises `ValueError` for invalid format and runtime errors for extraction failures.

#### `DocXLoader` (`rag_lib.loaders.docx`)

```python
DocXLoader(file_path: str)
```

- Converts DOCX to markdown.
- Returns one `Document` on success, `[]` on failures.
- Metadata includes `source_type="docx"`, `output_format="markdown"`.

#### `HTMLLoader` (`rag_lib.loaders.html`)

```python
HTMLLoader(
    file_path: str,
    output_format: Literal["markdown", "html"] = "markdown",
)
```

- Strict fail-fast loader (no fallback path).
- Returns exactly one `Document` on success.
- Raises on file read, parse, or render errors.
- Metadata includes `source_type="html"`, `output_format`.

#### `WebLoader` (`rag_lib.loaders.web`)

```python
WebLoader(
    url: str,
    depth: int = 0,
    output_format: Literal["markdown", "html"] = "markdown",
    fetch_mode: Literal["requests", "requests_fallback_playwright", "playwright"] = "requests",
    crawl_scope: Literal["same_host", "same_domain", "allowed_domains", "allow_all"] = "same_host",
    allowed_domains: Optional[List[str]] = None,
    login_url: Optional[str] = None,
    login_processor: Optional[Callable[[page, context, start_url, login_url, current_url], bool | None]] = None,
    follow_download_links: bool = False,
    request_timeout_seconds: float = 20.0,
    playwright_timeout_ms: int = 30000,
    playwright_headless: bool = True,
    ignore_https_errors: bool = False,
    user_agent: str = "rag-lib-webloader/1.0",
    max_pages: Optional[int] = None,
    retry_attempts: int = 1,
    continue_on_error: bool = True,
    cleanup_config: Optional[WebCleanupConfig] = None,
    custom_link_extractors: Optional[Sequence[Callable[[document, base_url], Sequence[WebLinkInput] | WebLinkInput | None]]] = None,
    playwright_link_extractor: Optional[Callable[[page, base_url], Sequence[WebLinkInput] | WebLinkInput | None]] = None,
    playwright_visible: Optional[bool] = None,
    playwright_extraction_config: Optional[PlaywrightExtractionConfig] = None,
    playwright_navigation_config: Optional[PlaywrightNavigationConfig] = None,
)
```

Behavior summary:

- Inclusive depth model: start URL is depth `0`.
- Regular/content links recurse to `depth + 1`.
- Navigation links recurse at the same depth.
- `non_recursive_classes` blocks regular links only (navigation links bypass this filter).
- Returns one `Document` per crawled HTML page or per Playwright navigation state, depending on navigation mode.
- Optional download routing (`follow_download_links=True`) routes downloadable responses to typed loaders.
- Login callback runs only for Playwright requests and reuses the same browser context.
- Diagnostics are available in `last_errors` and `last_stats`.

Parameter reference:

- `url`: crawl entrypoint.
- `depth`: max inclusive depth. `0` means crawl only the entry URL.
- `output_format`: rendered HTML output (`"markdown"` or `"html"`).
- `fetch_mode`:
  - `"requests"`: plain HTTP fetches only.
  - `"requests_fallback_playwright"`: requests first, then Playwright fallback on fetch failure/401/403; parse fallback paths also use Playwright.
  - `"playwright"`: browser-only fetch.
- `crawl_scope`:
  - `"same_host"`: exact host match.
  - `"same_domain"`: registrable-domain match.
  - `"allowed_domains"`: host must match `allowed_domains`.
  - `"allow_all"`: no host restriction.
- `allowed_domains`: host/domain allowlist when `crawl_scope="allowed_domains"`.
- `login_url`: optional auth URL hint used to detect login pages.
- `login_processor`: callback `(page, context, start_url, login_url, current_url) -> bool | None`.
  - `False` means login failed.
  - `True`/`None` means login flow accepted and crawl continues.
- `follow_download_links`: if enabled, downloadable resources are routed to existing loaders.
- `request_timeout_seconds`: per-request timeout for requests backend.
- `playwright_timeout_ms`: page navigation timeout for Playwright.
- `playwright_headless`: Playwright browser mode.
- `ignore_https_errors`: disables TLS cert validation in requests backend and sets Playwright `ignore_https_errors=True` (insecure workaround).
- `user_agent`: requests and Playwright user-agent value.
- `max_pages`: optional hard limit for number of visited URLs.
- `retry_attempts`: retry count per URL and backend.
- `continue_on_error`:
  - `True`: keep crawling and record errors.
  - `False`: raise on first recoverable failure.
- `cleanup_config`: optional content cleanup/link-filter pipeline:
  - `ignored_classes`: remove matching nodes before render/extraction.
  - `non_recursive_classes`: skip recursion for regular links with matching source classes.
  - `navigation_classes`: class-based navigation sources.
  - `navigation_styles`: style-snippet navigation sources.
  - `navigation_texts`: text-marker navigation sources (for example `"<"`, `">"`, `"next"`).
  - `duplicate_tags`: tag names participating in global duplicate filtering.
- `custom_link_extractors`: parsed-DOM callbacks run after cleanup.
- `playwright_link_extractor`: legacy Playwright page callback.
- `playwright_visible`: alias for headful mode (`True` forces `playwright_headless=False`).
- `playwright_extraction_config`: declarative Playwright extraction profile chain.
- `playwright_navigation_config`: generic Playwright navigation runner config.

Supporting config parameter reference:

- `WebLinkInput` accepted by custom/playwright extractors:
  - `str`: URL.
  - `WebLink`: structured URL + metadata.
  - `tuple[str, Sequence[str]]`: URL + source classes.
- `PlaywrightExtractionConfig` fields:
  - `profiles`: ordered `PlaywrightProfileConfig` sequence.
  - `continue_on_error`: if `True`, profile errors are recorded and execution continues.
  - `max_profile_runtime_ms`: optional wall-clock cap for full profile chain.
- `PlaywrightProfileConfig` fields:
  - `profile`: extraction mode (`anchors`, `attributes`, `onclick_regex`, `eval`, `paginated_eval`).
  - `selectors`: CSS selectors for profile modes that read elements.
  - `attributes`: attribute names used by `attributes`/`onclick_regex`.
  - `regex_pattern`: regex for `onclick_regex` value extraction.
  - `url_template`: interpolation template for regex captures (`{value}` default).
  - `script`: JavaScript for `eval`.
  - `script_args`: script argument payload.
  - `seed_script`: optional pre-loop script for `paginated_eval`.
  - `next_page_script`: optional page-advance script for `paginated_eval`.
  - `extract_script`: required extraction script for `paginated_eval`.
  - `max_pages`: max iterations for `paginated_eval`.
  - `wait_after_action_ms`: wait after script actions/page changes.
  - `include_url_patterns`: allowlist regex patterns applied to extracted URLs.
  - `exclude_url_patterns`: denylist regex patterns applied to extracted URLs.
  - `is_navigation`: marks extracted links as same-depth navigation links.
  - `source_tag`: metadata tag hint for extracted links.
  - `source_classes`: metadata class tokens for extracted links.
- `PlaywrightNavigationConfig` fields:
  - `enabled`: enable/disable generic cleanup-driven navigation runner.
  - `max_clicks`: max click attempts per page.
  - `max_states`: max captured states per page.
  - `wait_after_click_ms`: settle wait after a successful click.
  - `state_change_timeout_ms`: timeout waiting for content hash change.
  - `state_poll_interval_ms`: poll interval while waiting for state change.
  - `max_no_change_clicks`: stop after this many no-change clicks.
  - `clickable_selectors`: candidate selectors for click targets.
  - `forward_text_markers`: preferred forward labels/tokens.
  - `backward_text_markers`: de-prioritized backward labels/tokens.
  - `content_ready_selectors`: optional readiness selectors checked after change.
  - `navigation_state_document_mode`: state rendering mode (`separate_documents` or `single_document`).

Metadata and diagnostics produced by web crawl:

- HTML docs: `source`, `source_type="web"`, `output_format`, `web_depth`, `parent_url`, `fetch_backend`, `start_url`.
- Navigation-state docs also include `web_navigation_state_index`, `web_navigation_state_count`, `web_navigation_click_count`, `web_navigation_state_hash`, and `canonical_source`.
- Download-routed docs include `source_type="web_download"`, `download_content_type`, `download_filename`, `routed_loader`.
- `last_stats` keys: `visited_count`, `success_count`, `error_count`, `skipped_count`, `max_depth_reached`.
- `last_errors` entries include `stage` (`fetch`, `parse`, `download`, `auth`, `filter`) and backend details.

Playwright extraction behavior:

- Without `playwright_extraction_config`:
  - link discovery is mostly HTML-anchor based (`<a href>`) plus optional legacy callback.
  - useful for static pages and standard links.
- With `playwright_extraction_config`:
  - profile chain runs first, then `playwright_link_extractor`.
  - links are merged/deduped and can carry `is_navigation`, `source_tag`, `source_classes`.
  - supports dynamic extraction strategies (`attributes`, `onclick_regex`, `eval`, `paginated_eval`).

#### `AsyncWebLoader` (`rag_lib.loaders.web_async`)

```python
AsyncWebLoader(
    url: str,
    depth: int = 0,
    output_format: Literal["markdown", "html"] = "markdown",
    fetch_mode: Literal["requests", "requests_fallback_playwright", "playwright"] = "requests",
    crawl_scope: Literal["same_host", "same_domain", "allowed_domains", "allow_all"] = "same_host",
    allowed_domains: Optional[List[str]] = None,
    login_url: Optional[str] = None,
    login_processor: Optional[Callable[[page, context, start_url, login_url, current_url], Awaitable[bool | None] | bool | None]] = None,
    follow_download_links: bool = False,
    request_timeout_seconds: float = 20.0,
    playwright_timeout_ms: int = 30000,
    playwright_headless: bool = True,
    ignore_https_errors: bool = False,
    user_agent: str = "rag-lib-webloader/1.0",
    max_pages: Optional[int] = None,
    retry_attempts: int = 1,
    max_concurrency: int = 5,
    continue_on_error: bool = True,
    cleanup_config: Optional[WebCleanupConfig] = None,
    custom_link_extractors: Optional[Sequence[Callable[[document, base_url], Awaitable[Sequence[WebLinkInput]] | Sequence[WebLinkInput] | WebLinkInput | Awaitable[WebLinkInput] | None]]] = None,
    playwright_link_extractor: Optional[Callable[[page, base_url], Awaitable[Sequence[WebLinkInput]] | Sequence[WebLinkInput] | WebLinkInput | Awaitable[WebLinkInput] | None]] = None,
    playwright_visible: Optional[bool] = None,
    playwright_extraction_config: Optional[PlaywrightExtractionConfig] = None,
    playwright_navigation_config: Optional[PlaywrightNavigationConfig] = None,
)
```

Async-specific behavior:

- Same parameters and semantics as `WebLoader`, except:
  - `login_processor` may be sync or async.
  - `custom_link_extractors` and `playwright_link_extractor` may be sync or async.
  - `max_concurrency` controls concurrent URL processing.
- Crawl execution uses depth waves:
  - process current depth concurrently;
  - exhaust same-depth navigation waves;
  - then advance to `depth + 1` with regular links.

Examples:

1. Basic static crawl (`requests` only):

```python
loader = WebLoader(
    url="https://example.com",
    depth=1,
    fetch_mode="requests",
    output_format="markdown",
)
docs = loader.load()
```

2. Requests first, Playwright fallback, cleanup rules:

```python
loader = WebLoader(
    url="https://quotes.toscrape.com",
    depth=2,
    fetch_mode="requests_fallback_playwright",
    cleanup_config=WebCleanupConfig(
        non_recursive_classes=("tag",),
        navigation_classes=("pager",),
        ignored_classes=("footer",),
    ),
)
docs = loader.load()
```

3. Playwright extraction profiles enabled (dynamic links):

```python
profile = PlaywrightProfileConfig(
    profile="attributes",
    selectors=("[data-url]",),
    attributes=("data-url",),
)
loader = AsyncWebLoader(
    url="https://site-with-js-links.example",
    depth=1,
    fetch_mode="playwright",
    playwright_extraction_config=PlaywrightExtractionConfig(profiles=(profile,)),
)
docs = await loader.load()
```

4. Playwright extraction disabled (default):

```python
loader = AsyncWebLoader(
    url="https://site-with-js-links.example",
    depth=1,
    fetch_mode="playwright",
)
# This path relies on HTML anchors and cleanup/navigation signals only.
docs = await loader.load()
```

#### `CSVLoader` (`rag_lib.loaders.csv_excel`)

```python
CSVLoader(
    file_path: str,
    output_format: Literal["markdown", "csv"] = "markdown",
    delimiter: Optional[str] = None,
)
```

- Returns one `Document`.
- Auto-detects delimiter when omitted.
- Metadata includes `row_count`, `delimiter`, `output_format`.

#### `ExcelLoader` (`rag_lib.loaders.csv_excel`)

```python
ExcelLoader(
    file_path: str,
    output_format: Literal["markdown", "csv"] = "markdown",
    delimiter: str = ",",
    summarizer: Optional[TableSummarizer] = None,
)
```

- Returns one `Document` per sheet.
- Metadata includes `sheet_name`, `row_count`, `output_format`.

#### `RegexHierarchyLoader` (`rag_lib.loaders.regex`)

```python
RegexHierarchyLoader(
    file_path: str,
    patterns: Union[List[Tuple[int, str]], List[Dict]],
    exclude_patterns: Optional[List[str]] = None,
    include_parent_content: Union[bool, int] = False,
)
```

- Current behavior: loader returns raw file text as a single `Document`.
- Hierarchical splitting is done by `RegexHierarchySplitter`, not this loader.

#### `TableLoader` (`rag_lib.loaders.data_loaders`)

```python
TableLoader(file_path: str)
```

- Reads CSV-like data and renders one markdown table `Document`.
- Returns `[]` on failures.

#### `JsonLoader` (`rag_lib.loaders.data_loaders`)

```python
JsonLoader(
    file_path: str,
    output_format: Literal["json", "markdown"] = "json",
    schema: str = ".",
    schema_dialect: SchemaDialect = SchemaDialect.DOT_PATH,
    ensure_ascii: bool = False,
)
```

- Supports schema selection with dot-path traversal.
- Returns one `Document` or `[]` when path cannot be resolved / parsing fails.

#### `TextLoader` (`rag_lib.loaders.data_loaders`)

```python
TextLoader(file_path: str)
```

- Returns one plain text `Document` or `[]` on failure.

#### `MinerULoader` (`rag_lib.loaders.miner_u`)

```python
MinerULoader(
    file_path: str,
    parse_mode: str = "auto",          # "auto" | "txt" | "ocr"
    backend: Optional[str] = None,      # see enum below
    lang: Optional[str] = None,
    server_url: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    parse_formula: Optional[bool] = None,
    parse_table: Optional[bool] = None,
    device: Optional[str] = None,
    vram: Optional[int] = None,
    source: Optional[str] = None,       # "huggingface" | "modelscope" | "local"
    timeout_seconds: int = 600,
    keep_temp_artifacts: bool = False,
)
```

- Returns one markdown `Document` per PDF.
- Validates parameter ranges and enum-like values.
- Raises `ImportError` if MinerU is unavailable.

### 4.2 Loader enums

#### `SchemaDialect` (`rag_lib.loaders.data_loaders`)

```python
SchemaDialect = {"dot_path"}
```

#### Web crawl literals (`rag_lib.loaders.web`, `rag_lib.loaders.web_async`)

```python
FetchMode = {"requests", "requests_fallback_playwright", "playwright"}
CrawlScope = {"same_host", "same_domain", "allowed_domains", "allow_all"}
output_format = {"markdown", "html"}  # for WebLoader/AsyncWebLoader/HTMLLoader
```

#### Playwright extraction literals (`rag_lib.loaders.web_playwright_extractors`)

```python
PlaywrightProfileName = {"anchors", "attributes", "onclick_regex", "eval", "paginated_eval"}
NavigationStateDocumentMode = {"separate_documents", "single_document"}
```

`PlaywrightProfileName` value meaning:

- `anchors`: extract URLs from selected anchors (`href`).
- `attributes`: extract URLs from configured attributes (for example `data-url`, `data-href`).
- `onclick_regex`: extract URL-like values from attributes/text via regex.
- `eval`: evaluate one custom JavaScript extraction script.
- `paginated_eval`: iterative extraction across pages/states via `seed_script`, `extract_script`, `next_page_script`.

`NavigationStateDocumentMode` value meaning:

- `separate_documents`: emit one `Document` per captured navigation state (`#nav-state=N` source suffix).
- `single_document`: merge all captured states into one combined document.

#### Download routing kinds (internal mapping in `web_common`)

```python
download_kind = {"pdf", "docx", "html", "csv", "xlsx", "json", "txt"}
```

Default routed loaders by kind:

- `pdf` -> `PDFLoader` (fallback: `PyMuPDFLoader`)
- `docx` -> `DocXLoader`
- `html` -> `HTMLLoader`
- `csv` -> `CSVLoader`
- `xlsx` -> `ExcelLoader`
- `json` -> `JsonLoader`
- `txt` -> `TextLoader`

## 5. Chunkers and Splitters (`rag_lib.chunkers`)

### 5.1 Base contract

`TextSplitter` base class exposes:

- `split_text(text: str) -> List[str]`
- `create_segments(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]`
- `split_documents(documents: Iterable[Document]) -> List[Segment]`
- `split_segments(segments: Iterable[Segment]) -> List[Segment]`

### 5.2 Concrete classes and parameter signatures

#### `SemanticChunker`

```python
SemanticChunker(
    embeddings: Embeddings,
    threshold: Optional[float] = None,
    window_size: int = 1,
    language: str = "auto",
    threshold_type: str = "fixed",          # "fixed" | "percentile" | "percentile_local"
    percentile_threshold: int = 90,
    local_percentile_window: int = 50,
    local_min_samples: int = 20,
    local_fallback: str = "global",         # "global" | "fixed"
    enable_debug: bool = False,
)
```

- Adds debug accessors: `get_last_debug_info()` and `get_split_boundary_for_chunk(...)`.

#### `RecursiveCharacterTextSplitter`

```python
RecursiveCharacterTextSplitter(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
    separators: Optional[List[str]] = None,
    keep_separator: bool = False,
    is_separator_regex: bool = False,
)
```

#### `TokenTextSplitter`

```python
TokenTextSplitter(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
    model_name: str = "cl100k_base",
    encoding_name: Optional[str] = None,
)
```

#### `SentenceSplitter`

```python
SentenceSplitter(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
    language: str = "auto",
)
```

#### `RegexSplitter`

```python
RegexSplitter(
    pattern: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable[[str], int] = len,
)
```

#### `RegexHierarchySplitter`

```python
RegexHierarchySplitter(
    patterns: Union[List[Tuple[int, str]], List[Dict]],
    exclude_patterns: Optional[List[str]] = None,
    include_parent_content: Union[bool, int] = False,
)
```

#### `MarkdownHierarchySplitter`

```python
MarkdownHierarchySplitter(
    exclude_code_blocks: bool = True,
    include_parent_content: Union[bool, int] = False,
)
```

#### `JsonSplitter`

```python
JsonSplitter(
    min_chunk_size: int = 0,
    schema: str = ".",
    schema_dialect: SchemaDialect = SchemaDialect.DOT_PATH,
    ensure_ascii: bool = False,
    metadata_value_max_len: Optional[int] = 256,
)
```

#### `QASplitter`

```python
QASplitter()
```

#### `MarkdownTableSplitter`

```python
MarkdownTableSplitter(
    *,
    split_table_rows: bool = False,
    use_first_row_as_header: bool = True,
    max_rows_per_chunk: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    summarizer: Optional[TableSummarizer] = None,
    summarize_table: bool = True,
    summarize_chunks: bool = False,
    inject_summaries_into_content: bool = False,
)
```

#### `CSVTableSplitter`

```python
CSVTableSplitter(
    max_rows_per_chunk: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    delimiter: Optional[str] = None,
    use_first_row_as_header: bool = True,
    summarizer: Optional[TableSummarizer] = None,
    summarize_table: bool = True,
    summarize_chunks: bool = False,
    inject_summaries_into_content: bool = False,
    length_function: Callable[[str], int] = len,
)
```

#### `HTMLSplitter`

```python
HTMLSplitter(
    output_format: Literal["markdown", "html"] = "markdown",
    split_table_rows: bool = False,
    use_first_row_as_header: bool = True,
    max_rows_per_chunk: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    summarizer: Optional[TableSummarizer] = None,
    summarize_table: bool = True,
    summarize_chunks: bool = False,
    inject_summaries_into_content: bool = False,
    include_parent_content: Union[bool, int] = False,
)
```

- Splits by HTML heading hierarchy (`h1..h6`), preserving lists and tables.
- Emits separate `SegmentType.TABLE` chunks for `<table>`.
- Supports row chunking and summary metadata parity with markdown/csv table splitters.
- Strict fail-fast behavior: parse/render/table/summarizer errors are raised.

## 6. Processors and Indexing

### 6.1 Processors (`rag_lib.processors`)

#### `SegmentEnricher`

```python
SegmentEnricher(llm: BaseChatModel)
```

- `enrich(segments: List[Segment]) -> List[Segment]`
- `aenrich(segments: List[Segment]) -> List[Segment]`
- Injects `generated_title`, `keywords`, and `summary` into metadata.

#### `EntityExtractor`

```python
EntityExtractor(llm: BaseChatModel, store: BaseGraphStore)
```

- `process_segments(segments: List[Segment]) -> None`
- `aprocess_segments(segments: List[Segment], concurrency: int = 5) -> None`

#### `CommunitySummarizer`

```python
CommunitySummarizer(llm: BaseChatModel, store: BaseGraphStore)
```

- `summarize(communities: Dict[int, List[str]]) -> List[Segment]`

#### `RaptorProcessor`

```python
RaptorProcessor(
    llm: BaseChatModel,
    embeddings: Embeddings,
    max_levels: int = 3,
    clustering_service: Optional[ClusteringService] = None,
    summary_prompt_template: str | None = None,
)
```

- `process_segments(segments: List[Segment]) -> List[Segment]`
- `aprocess_segments(segments: List[Segment]) -> List[Segment]`

### 6.2 `Indexer` (`rag_lib.core.indexer`)

```python
Indexer(
    vector_store: VectorStore,
    embeddings: Embeddings,
    enricher: Optional[SegmentEnricher] = None,
    entity_extractor: Optional[EntityExtractor] = None,
    doc_store: Optional[BaseStore[str, Any]] = None,
)
```

Methods:

- `index(segments: List[Segment], parent_segments: Optional[List[Segment]] = None, batch_size: int = 100) -> None`
- `aindex(segments: List[Segment], parent_segments: Optional[List[Segment]] = None, batch_size: int = 100) -> None`

Behavior:

- Optional enrichment before indexing.
- Optional graph extraction during indexing.
- Optional dual-storage hydration support via `doc_store`.

## 7. Retrieval (`rag_lib.retrieval`)

### 7.1 Atomic retrievers/factories (`retrievers.py`)

```python
RegexRetriever(documents: List[Union[Document, Segment]])
FuzzyRetriever(documents: List[Union[Document, Segment]], threshold: int = 80, mode: str = "partial_ratio")

create_vector_retriever(
    vector_store: VectorStore,
    top_k: int = 4,
    search_type: str = "similarity",      # similarity | mmr | similarity_score_threshold
    score_threshold: Optional[float] = None,
) -> BaseRetriever

create_bm25_retriever(
    documents: List[Union[Document, Segment]],
    top_k: int = 4,
) -> BM25Retriever

create_graph_retriever(
    vector_store: Optional[VectorStore],
    graph_store: Any,
    config: Optional[Any] = None,
    embedder: Optional[Embeddings] = None,
    llm: Optional[BaseChatModel] = None,
    doc_store: Optional[BaseStore[str, Document]] = None,
    id_key: str = "segment_id",
) -> BaseRetriever
```

### 7.2 Composition factories (`composition.py`)

```python
create_ensemble_retriever(retrievers: List[BaseRetriever], weights: Optional[List[float]] = None) -> BaseRetriever

create_dual_storage_retriever(
    vector_store: VectorStore,
    doc_store: BaseStore[str, Document],
    id_key: str = "segment_id",
    search_kwargs: Optional[Dict[str, Any]] = None,
) -> MultiVectorRetriever

create_scored_dual_storage_retriever(
    vector_store: VectorStore,
    doc_store: BaseStore[str, Document],
    id_key: str = "segment_id",
    search_kwargs: Optional[Dict[str, Any]] = None,
    search_type: SearchType = SearchType.similarity,
    score_threshold: float | None = None,
    hydration_mode: HydrationMode = HydrationMode.parents_replace,
    enrichment_separator: str = "\n\n--- MATCHED CHILD CHUNK ---\n\n",
) -> BaseRetriever

create_reranking_retriever(
    base_retriever_or_list: Union[BaseRetriever, List[BaseRetriever]],
    reranker_model: Union[str, BaseCrossEncoder] = "BAAI/bge-reranker-base",
    top_k: int = 5,
    max_score_ratio: float = 0.0,
    device: str = "cpu",
) -> ContextualCompressionRetriever

create_graph_hybrid_retriever(
    vector_retriever: BaseRetriever,
    graph_retriever: BaseRetriever,
    weights: List[float] = [0.7, 0.3],
) -> EnsembleRetriever
```

### 7.3 `ScoredMultiVectorRetriever` enums and defaults (`scored_retriever.py`)

#### `SearchType`

```python
SearchType = {
    "similarity",
    "similarity_score_threshold",
    "mmr",
}
```

#### `HydrationMode`

```python
HydrationMode = {
    "parents_replace",
    "children_enriched",
    "children_plus_parents",
}
```

#### Constructor fields

```python
ScoredMultiVectorRetriever(
    vector_store: VectorStore,
    byte_store: ByteStore | None = None,
    doc_store: BaseStore[str, Document],
    id_key: str = "doc_id",
    search_kwargs: dict = {},
    search_type: SearchType = SearchType.similarity,
    score_threshold: float | None = None,
    hydration_mode: HydrationMode = HydrationMode.parents_replace,
    enrichment_separator: str = "\n\n--- MATCHED CHILD CHUNK ---\n\n",
)
```

Note: factory `create_scored_dual_storage_retriever(...)` defaults `id_key` to `"segment_id"`, overriding the class-level default.

## 8. Graph RAG

### 8.1 Graph domain (`rag_lib.graph`)

#### `GraphNode`

```python
GraphNode(
    id: str,
    type: str,
    label: str,
    description: Optional[str] = None,
    properties: Dict[str, Any] = {},
    source_segment_id: Optional[str] = None,
)
```

#### `GraphEdge`

```python
GraphEdge(
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float = 1.0,
    properties: Dict[str, Any] = {},
    source_segment_id: Optional[str] = None,
)
```

Stores:

- `NetworkXGraphStore()` (in-memory)
- `Neo4jGraphStore(uri: str, auth: tuple[str, str], database: str = "neo4j")`

### 8.2 `GraphQueryConfig` (`rag_lib.retrieval.graph_retriever`)

```python
GraphQueryConfig(
    mode: Literal["local", "global", "hybrid", "mix"] = "hybrid",
    top_k_entities: int = 12,
    top_k_relations: int = 24,
    top_k_chunks: int = 10,
    max_hops: int = 2,
    min_score: float = 0.15,
    use_rerank: bool = True,
    enable_keyword_extraction: bool = True,
    vector_relevance_mode: Literal["strict_0_1", "normalize_minmax"] = "strict_0_1",
    token_budget_total: int = 3500,
    token_budget_entities: int = 700,
    token_budget_relations: int = 900,
    token_budget_chunks: int = 1900,
)
```

### 8.3 `GraphRetriever` constructor fields

```python
GraphRetriever(
    vector_store: Optional[VectorStore] = None,
    graph_store: BaseGraphStore,
    config: GraphQueryConfig = GraphQueryConfig(),
    embedder: Optional[Embeddings] = None,
    llm: Optional[BaseChatModel] = None,
    doc_store: Optional[BaseStore[str, Document]] = None,
    id_key: str = "segment_id",
)
```

Runtime contract:

- `vector_store` is required in strict mode (validated at runtime).
- If `enable_keyword_extraction=True`, `llm` with `with_structured_output(...)` is required.

Public methods:

- `retrieve(query: str, top_k: Optional[int] = None) -> List[Document]`
- `aretrieve(query: str, top_k: Optional[int] = None) -> List[Document]`
- `.invoke(...)` / `.ainvoke(...)` are supported via `BaseRetriever`.

### 8.4 Graph retriever returned metadata contract

Each returned `Document` includes:

- `retrieval_kind`: one of `chunk`, `entity`, `relation`, `community`
- `score`: normalized float in `[0.0, 1.0]`
- `graph_mode`: one of `local`, `global`, `hybrid`, `mix`
- `source_segment_id` when available
- `entity_id` for entity hits
- `edge_id` for relation hits
- `community_id` for community hits

### 8.5 Graph retriever strict exceptions

- `GraphConfigurationError`
- `GraphCapabilityError`
- `GraphDataError`

## 9. MinerU Detailed Parameters and Enums

### 9.1 `parse_mode` enum-like values

```python
{"auto", "txt", "ocr"}
```

### 9.2 `backend` enum-like values

```python
{
  "pipeline",
  "hybrid-auto-engine",
  "hybrid-http-client",
  "vlm-auto-engine",
  "vlm-http-client",
}
```

### 9.3 `source` enum-like values

```python
{"huggingface", "modelscope", "local"}
```

### 9.4 Return metadata keys from `MinerULoader.load()`

Always present:

- `source`
- `parser = "MinerU"`
- `output_format = "markdown"`
- `mineru_parse_mode`
- `mineru_command`

Conditionally present:

- `mineru_backend`
- `lang`
- `mineru_source`
- `start_page`
- `end_page`

## 10. RAPTOR (`rag_lib.raptor` + `rag_lib.processors.raptor`)

### 10.1 Components and current constructor parameters

#### `ClusteringService`

```python
ClusteringService()
```

#### `ClusterSummarizer`

```python
ClusterSummarizer(
    llm: BaseChatModel,
    summary_prompt_template: str | None = None,
)
```

Methods:

- `summarize(texts: List[str], target_language="english", max_chars=1200, target_ratio=0.35) -> str`
- `asummarize(...) -> str`

#### `TreeBuilder`

```python
TreeBuilder(
    clustering_service: ClusteringService,
    summarizer: ClusterSummarizer,
    embeddings_model: Embeddings,
    summary_target_ratio: float = 0.35,
    summary_max_chars: int = 1200,
    summary_min_chars: int = 120,
    summary_preserve_language: bool = True,
    strict_quality: bool = True,
)
```

Methods:

- `build(segments: List[Segment], n_levels: int = 3) -> List[Segment]`
- `abuild(segments: List[Segment], n_levels: int = 3) -> List[Segment]`

### 10.2 RAPTOR segment metadata emitted by `TreeBuilder`

Summary segments include keys such as:

- `raptor_level`
- `raptor_cluster_id`
- `raptor_child_ids`
- `is_raptor_summary`
- `raptor_summary_chars`
- `raptor_children_chars`
- `raptor_compression_ratio`
- `raptor_summary_language`
- `raptor_summary_max_chars`
- `raptor_summary_exceeds_max_chars`

After hierarchy finalization additional keys may include:

- `raptor_parent_ids`
- `raptor_depth_from_root`

## 11. Configuration (`rag_lib.config`)

`Settings` structure:

- `llm: LLMSettings` (`LLM_` prefix)
- `embeddings: EmbeddingsSettings` (`EMBEDDING_` prefix)
- `vector_store: VectorStoreSettings` (`VECTOR_` prefix)
- `ingestion: IngestionSettings` (`INGEST_` prefix)
- `prompts: PromptSettings` (`PROMPT_` prefix)
- top-level: `log_level`, `openai_api_key`, `openai_api_key_personal`, `mistral_api_key`, `ya_api_key`, `ya_folder_id`

Notable defaults:

- `ingestion.chunk_size = 100`
- `ingestion.semantic_threshold = 0.6`
- `ingestion.default_pdf_backend = "poppler"`

## 12. Corrected Usage Examples

### 12.1 MinerU PDF to chunks

```python
from rag_lib.loaders.miner_u import MinerULoader
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter

loader = MinerULoader(
    "complex_layout.pdf",
    parse_mode="txt",
    start_page=0,
    end_page=4,
    parse_formula=False,
    parse_table=False,
    timeout_seconds=1200,
)
docs = loader.load()  # List[Document]

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
segments = splitter.split_documents(docs)  # List[Segment]
```

### 12.2 Graph retrieval (strict lexical keywords, no LLM keyword extraction)

```python
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphRetriever, GraphQueryConfig

graph_store = NetworkXGraphStore()
# Populate graph_store + vector_store first.

retriever = GraphRetriever(
    vector_store=vector_store,
    graph_store=graph_store,
    config=GraphQueryConfig(
        mode="hybrid",
        enable_keyword_extraction=False,
        top_k_entities=8,
        top_k_relations=10,
        top_k_chunks=8,
    ),
)

docs = retriever.invoke("probability theory")
```

### 12.3 Scored dual storage retriever with explicit hydration mode

```python
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType, HydrationMode

retriever = create_scored_dual_storage_retriever(
    vector_store=vector_store,
    doc_store=doc_store,
    id_key="segment_id",
    search_type=SearchType.similarity_score_threshold,
    hydration_mode=HydrationMode.parents_replace,
    score_threshold=0.4,
)

docs = retriever.invoke("query")
```

## 13. Migration Notes

- `rag_lib.loaders.structured` is removed and raises `ImportError`.
- Use `from rag_lib.loaders.docx import DocXLoader`.
- `RegexHierarchyLoader` is now a raw text loader; hierarchical splitting belongs to `RegexHierarchySplitter` / `MarkdownHierarchySplitter`.
- `HTMLLoader` and `HTMLSplitter` are strict (no fallback behavior for malformed input).
- `rag_lib.__version__` is not defined in package code; use `importlib.metadata.version("rag-lib")`.
