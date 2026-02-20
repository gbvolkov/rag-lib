from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Tuple
from urllib.parse import urlparse

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger
from rag_lib.loaders.web_common import (
    WebCleanupConfig,
    WebLink,
    WebLinkInput,
    cleanup_and_extract_web_links,
    get_header,
    is_download_response,
    is_html_response,
    is_url_in_scope,
    merge_web_links,
    normalize_content_type,
    normalize_url,
    normalize_web_link_input,
    parse_web_html_document,
    partition_web_links,
    render_web_html_document,
    route_download_content_to_documents,
)
from rag_lib.loaders.web_playwright_extractors import (
    NavigationStateDocumentMode,
    PlaywrightExtractionConfig,
    PlaywrightNavigationConfig,
    PlaywrightNavigationState,
    run_async_cleanup_navigation,
    run_async_playwright_extraction,
)

FetchMode = Literal["requests", "requests_fallback_playwright", "playwright"]
CrawlScope = Literal["same_host", "same_domain", "allowed_domains", "allow_all"]


@dataclass
class _FetchResult:
    backend: str
    url: str
    final_url: str
    status_code: Optional[int] = None
    content_type: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    content_bytes: bytes = b""
    extra_links: List[WebLink] = field(default_factory=list)
    extractor_errors: List[str] = field(default_factory=list)
    navigation_states: List[PlaywrightNavigationState] = field(default_factory=list)
    navigation_click_count: int = 0
    error: str = ""


@dataclass
class _ProcessResult:
    documents: List[Document] = field(default_factory=list)
    regular_links: List[WebLink] = field(default_factory=list)
    navigation_links: List[WebLink] = field(default_factory=list)
    skipped_count: int = 0
    source_url: Optional[str] = None
    backend: str = ""


class AsyncWebLoader:
    """
    Asynchronous recursive web crawler with bounded concurrency and the same feature surface as
    `WebLoader` (cleanup filters, Playwright extraction/navigation, download routing, diagnostics).

    Depth semantics are inclusive:
    - start URL is depth `0`;
    - regular/content links recurse to `depth + 1`;
    - navigation links recurse at the same depth;
    - `non_recursive_classes` applies to regular links only.

    Async traversal processes each depth in waves:
    - same-depth navigation waves are exhausted first;
    - regular links advance to the next depth frontier.

    Diagnostics:
    - `last_errors`: list of fetch/parse/auth/download/filter diagnostics;
    - `last_stats`: aggregate crawl counters (`visited_count`, `success_count`, `error_count`,
      `skipped_count`, `max_depth_reached`).
    """

    def __init__(
        self,
        url: str,
        depth: int = 0,
        output_format: Literal["markdown", "html"] = "markdown",
        fetch_mode: FetchMode = "requests",
        crawl_scope: CrawlScope = "same_host",
        allowed_domains: Optional[List[str]] = None,
        login_url: Optional[str] = None,
        login_processor: Optional[
            Callable[[Any, Any, str, Optional[str], str], Awaitable[bool | None] | bool | None]
        ] = None,
        follow_download_links: bool = False,
        request_timeout_seconds: float = 20.0,
        playwright_timeout_ms: int = 30_000,
        playwright_headless: bool = True,
        ignore_https_errors: bool = False,
        user_agent: str = "rag-lib-webloader/1.0",
        max_pages: Optional[int] = None,
        retry_attempts: int = 1,
        max_concurrency: int = 5,
        continue_on_error: bool = True,
        cleanup_config: WebCleanupConfig | None = None,
        custom_link_extractors: Sequence[
            Callable[
                [Any, str],
                Sequence[WebLinkInput] | Awaitable[Sequence[WebLinkInput]] | WebLinkInput | Awaitable[WebLinkInput] | None,
            ]
        ]
        | None = None,
        playwright_link_extractor: Optional[
            Callable[
                [Any, str],
                Sequence[WebLinkInput]
                | Awaitable[Sequence[WebLinkInput]]
                | WebLinkInput
                | Awaitable[WebLinkInput]
                | None,
            ]
        ] = None,
        playwright_visible: Optional[bool] = None,
        playwright_extraction_config: PlaywrightExtractionConfig | None = None,
        playwright_navigation_config: PlaywrightNavigationConfig | None = None,
    ):
        """
        Initialize an asynchronous web crawler.

        Args:
            url: Start URL.
            depth: Inclusive crawl depth (0 means only start URL).
            output_format: Rendered HTML document format (`"markdown"` or `"html"`).
            fetch_mode:
                - `"requests"`: HTTP client only.
                - `"requests_fallback_playwright"`: requests first; fallback to Playwright on fetch
                  failures/401/403 and on parse fallback paths.
                - `"playwright"`: browser-only retrieval.
            crawl_scope:
                - `"same_host"`: exact hostname match with start URL.
                - `"same_domain"`: registrable domain match.
                - `"allowed_domains"`: host must match `allowed_domains`.
                - `"allow_all"`: no host restriction.
            allowed_domains: Domain allowlist used when `crawl_scope="allowed_domains"`.
            login_url: Optional login URL hint used for login trigger detection.
            login_processor: Optional callback run in shared async Playwright context when auth is required.
                Signature: `(page, context, start_url, login_url, current_url) -> Awaitable[bool | None] | bool | None`.
            follow_download_links: Route downloadable responses through file loaders when true.
            request_timeout_seconds: Per-request timeout for requests backend.
            playwright_timeout_ms: Playwright navigation timeout for page loads.
            playwright_headless: Playwright headless mode.
            ignore_https_errors: Disables TLS certificate verification for requests backend and enables
                Playwright `ignore_https_errors`. Use only as an insecure workaround.
            user_agent: User-Agent header/browser context value.
            max_pages: Optional cap on visited URLs.
            retry_attempts: Retry count for each fetch attempt (0 means single attempt).
            max_concurrency: Maximum concurrent in-flight URL processing tasks.
            continue_on_error: Continue crawl on recoverable errors; otherwise raise immediately.
            cleanup_config: Optional DOM cleanup/link-filter config.
            custom_link_extractors: Extra parsed-DOM extractors run after cleanup.
            playwright_link_extractor: Legacy Playwright extractor callback.
            playwright_visible: Convenience alias for headful Playwright (`True` forces `playwright_headless=False`).
            playwright_extraction_config: Declarative Playwright profile chain executed before
                `playwright_link_extractor`.
            playwright_navigation_config: Generic Playwright navigation runner settings.
        """
        if not url:
            raise ValueError("url must be non-empty")
        if depth < 0:
            raise ValueError("depth must be >= 0")
        if output_format not in {"markdown", "html"}:
            raise ValueError("output_format must be 'markdown' or 'html'")
        if fetch_mode not in {"requests", "requests_fallback_playwright", "playwright"}:
            raise ValueError("fetch_mode must be requests, requests_fallback_playwright, or playwright")
        if crawl_scope not in {"same_host", "same_domain", "allowed_domains", "allow_all"}:
            raise ValueError("invalid crawl_scope")
        if max_pages is not None and max_pages <= 0:
            raise ValueError("max_pages must be > 0 when provided")
        if retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self.url = normalize_url(url)
        self.depth = depth
        self.output_format = output_format
        self.fetch_mode = fetch_mode
        self.crawl_scope = crawl_scope
        self.allowed_domains = allowed_domains or []
        self.login_url = normalize_url(login_url) if login_url else None
        self.login_processor = login_processor
        self.follow_download_links = follow_download_links
        self.request_timeout_seconds = request_timeout_seconds
        self.playwright_timeout_ms = playwright_timeout_ms
        self.playwright_headless = (not playwright_visible) if playwright_visible is not None else playwright_headless
        self.ignore_https_errors = ignore_https_errors
        self.user_agent = user_agent
        self.max_pages = max_pages
        self.retry_attempts = retry_attempts
        self.max_concurrency = max_concurrency
        self.continue_on_error = continue_on_error
        self.cleanup_config = cleanup_config
        self.custom_link_extractors = list(custom_link_extractors or [])
        self.playwright_extraction_config = playwright_extraction_config
        self.playwright_navigation_config = playwright_navigation_config or PlaywrightNavigationConfig()
        self.playwright_link_extractor = playwright_link_extractor

        self.last_errors: List[Dict[str, Any]] = []
        self.last_stats: Dict[str, Any] = {}

        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._playwright_driver: Any = None
        self._playwright_browser: Any = None
        self._playwright_context: Any = None
        self._playwright_init_lock = asyncio.Lock()
        self._login_lock = asyncio.Lock()
        self._login_completed = False

        self._duplicate_hashes: set[str] = set()
        self._navigation_hashes: set[str] = set()

    async def load(self) -> List[Document]:
        logger.info(
            "Async loading web content: %s depth=%s mode=%s scope=%s concurrency=%s",
            self.url,
            self.depth,
            self.fetch_mode,
            self.crawl_scope,
            self.max_concurrency,
        )

        self.last_errors = []
        self.last_stats = {}
        self._duplicate_hashes = set()
        self._navigation_hashes = set()

        documents: List[Document] = []
        visited: set[str] = set()
        enqueued_urls: set[str] = {self.url}
        skipped_count = 0
        max_depth_reached = 0

        frontier: List[Tuple[str, Optional[str]]] = [(self.url, None)]
        stop_crawl = False

        try:
            for current_depth in range(self.depth + 1):
                if stop_crawl or not frontier:
                    break

                max_depth_reached = max(max_depth_reached, current_depth)
                wave_frontier = frontier
                frontier = []

                while wave_frontier:
                    scheduled: List[Tuple[str, Optional[str]]] = []
                    for url, parent_url in wave_frontier:
                        if url in visited:
                            continue
                        if self.max_pages is not None and len(visited) >= self.max_pages:
                            stop_crawl = True
                            break
                        visited.add(url)
                        scheduled.append((url, parent_url))

                    if not scheduled:
                        break

                    tasks = [
                        asyncio.create_task(self._process_url(url=url, depth=current_depth, parent_url=parent_url))
                        for url, parent_url in scheduled
                    ]
                    results = await asyncio.gather(*tasks)

                    next_same_depth_wave: List[Tuple[str, Optional[str]]] = []
                    same_depth_seen: set[str] = set()

                    for result in results:
                        documents.extend(result.documents)
                        skipped_count += result.skipped_count

                        if not result.source_url:
                            continue

                        for nav_link in result.navigation_links:
                            if not is_url_in_scope(
                                nav_link.url,
                                self.url,
                                crawl_scope=self.crawl_scope,
                                allowed_domains=self.allowed_domains,
                            ):
                                skipped_count += 1
                                self._record_filter(
                                    nav_link.url,
                                    current_depth,
                                    f"URL out of crawl scope ({self.crawl_scope}).",
                                    result.backend,
                                )
                                continue
                            if (
                                nav_link.url not in visited
                                and nav_link.url not in enqueued_urls
                                and nav_link.url not in same_depth_seen
                            ):
                                next_same_depth_wave.append((nav_link.url, result.source_url))
                                same_depth_seen.add(nav_link.url)
                                enqueued_urls.add(nav_link.url)

                        if current_depth >= self.depth:
                            continue

                        for link in result.regular_links:
                            if self._is_non_recursive_link(link):
                                skipped_count += 1
                                self._record_filter(
                                    link.url,
                                    current_depth,
                                    "Skipped by non_recursive_classes.",
                                    result.backend,
                                )
                                continue
                            if not is_url_in_scope(
                                link.url,
                                self.url,
                                crawl_scope=self.crawl_scope,
                                allowed_domains=self.allowed_domains,
                            ):
                                skipped_count += 1
                                self._record_filter(
                                    link.url,
                                    current_depth,
                                    f"URL out of crawl scope ({self.crawl_scope}).",
                                    result.backend,
                                )
                                continue
                            if link.url not in visited and link.url not in enqueued_urls:
                                frontier.append((link.url, result.source_url))
                                enqueued_urls.add(link.url)

                    wave_frontier = next_same_depth_wave
                    if stop_crawl:
                        break

                if stop_crawl:
                    break

            self.last_stats = {
                "visited_count": len(visited),
                "success_count": len(documents),
                "error_count": sum(1 for err in self.last_errors if err.get("stage") != "filter"),
                "skipped_count": skipped_count,
                "max_depth_reached": max_depth_reached,
            }
            return documents
        finally:
            await self._close_resources()

    async def _process_url(self, *, url: str, depth: int, parent_url: Optional[str]) -> _ProcessResult:
        async with self._semaphore:
            result = await self._fetch_url(url)
            if result.error:
                self._record_error(url, depth, "fetch", result.error, result.backend)
                if not self.continue_on_error:
                    raise RuntimeError(result.error)
                return _ProcessResult(skipped_count=1, backend=result.backend)

            final_url = normalize_url(result.final_url or url)
            is_html = is_html_response(final_url, result.content_type)

            if is_html:
                source_url = final_url
                fetch_backend = result.backend

                try:
                    rendered_documents, regular_links, navigation_links = await self._build_documents_from_fetch_result(
                        fetch_result=result,
                        final_url=final_url,
                        depth=depth,
                        parent_url=parent_url,
                    )
                except Exception as exc:
                    fallback_result, fallback_error = await self._try_playwright_parse_fallback(
                        url=url,
                        primary_backend=result.backend,
                    )
                    if fallback_result is None:
                        message = str(exc)
                        if fallback_error:
                            message = f"{message} | {fallback_error}"
                        self._record_error(final_url, depth, "parse", message, result.backend)
                        if not self.continue_on_error:
                            raise
                        return _ProcessResult(skipped_count=1, source_url=final_url, backend=result.backend)

                    source_url = normalize_url(fallback_result.final_url or url)
                    fetch_backend = fallback_result.backend
                    try:
                        rendered_documents, regular_links, navigation_links = await self._build_documents_from_fetch_result(
                            fetch_result=fallback_result,
                            final_url=source_url,
                            depth=depth,
                            parent_url=parent_url,
                        )
                    except Exception as fallback_exc:
                        self._record_error(source_url, depth, "parse", str(fallback_exc), fallback_result.backend)
                        if not self.continue_on_error:
                            raise
                        return _ProcessResult(skipped_count=1, source_url=source_url, backend=fallback_result.backend)

                return _ProcessResult(
                    documents=rendered_documents,
                    regular_links=regular_links,
                    navigation_links=navigation_links,
                    source_url=source_url,
                    backend=fetch_backend,
                )

            if self.follow_download_links and is_download_response(
                final_url,
                result.content_type,
                result.headers,
            ):
                if not result.content_bytes:
                    error_msg = "No bytes available for downloadable response."
                    self._record_error(final_url, depth, "download", error_msg, result.backend)
                    if not self.continue_on_error:
                        raise RuntimeError(error_msg)
                    return _ProcessResult(skipped_count=1, source_url=final_url, backend=result.backend)

                try:
                    download_docs = await asyncio.to_thread(
                        route_download_content_to_documents,
                        content_bytes=result.content_bytes,
                        source_url=final_url,
                        content_type=result.content_type,
                        headers=result.headers,
                        output_format=self.output_format,
                    )
                except Exception as exc:
                    self._record_error(final_url, depth, "download", str(exc), result.backend)
                    if not self.continue_on_error:
                        raise
                    return _ProcessResult(skipped_count=1, source_url=final_url, backend=result.backend)

                return _ProcessResult(
                    documents=self._annotate_download_documents(
                        docs=download_docs,
                        depth=depth,
                        parent_url=parent_url,
                        backend=result.backend,
                    ),
                    source_url=final_url,
                    backend=result.backend,
                )

            return _ProcessResult(skipped_count=1, source_url=final_url, backend=result.backend)

    async def _build_documents_from_fetch_result(
        self,
        *,
        fetch_result: _FetchResult,
        final_url: str,
        depth: int,
        parent_url: str | None,
    ) -> tuple[list[Document], list[WebLink], list[WebLink]]:
        states = self._extract_html_states_from_fetch_result(fetch_result=fetch_result)
        return await self._build_documents_from_html_states(
            states=states,
            canonical_source=final_url,
            depth=depth,
            parent_url=parent_url,
            backend=fetch_result.backend,
            status_code=fetch_result.status_code,
        )

    def _extract_html_states_from_fetch_result(self, *, fetch_result: _FetchResult) -> list[PlaywrightNavigationState]:
        if fetch_result.navigation_states:
            return list(fetch_result.navigation_states)

        raw_html = fetch_result.text or (
            fetch_result.content_bytes.decode("utf-8", errors="replace") if fetch_result.content_bytes else ""
        )
        if not raw_html:
            raise ValueError("HTML response is empty.")

        return [
            PlaywrightNavigationState(
                html=raw_html,
                content_hash=hashlib.md5(raw_html.encode("utf-8")).hexdigest(),
                click_count=0,
                extra_links=tuple(fetch_result.extra_links),
                extractor_errors=tuple(fetch_result.extractor_errors),
            )
        ]

    async def _build_documents_from_html_states(
        self,
        *,
        states: Sequence[PlaywrightNavigationState],
        canonical_source: str,
        depth: int,
        parent_url: str | None,
        backend: str,
        status_code: Optional[int],
    ) -> tuple[list[Document], list[WebLink], list[WebLink]]:
        if not states:
            raise ValueError("No HTML states available.")

        mode: NavigationStateDocumentMode = self.playwright_navigation_config.navigation_state_document_mode
        state_count = len(states)
        documents: list[Document] = []
        combined_regular_links: list[WebLink] = []
        combined_navigation_links: list[WebLink] = []
        combined_content_parts: list[str] = []
        first_error: Exception | None = None

        for index, state in enumerate(states):
            state_source = self._state_source_url(
                canonical_source=canonical_source,
                state_index=index,
                state_count=state_count,
                mode=mode,
            )
            if state.extractor_errors:
                self._record_playwright_extractor_errors(
                    url=state_source,
                    depth=depth,
                    backend=backend,
                    messages=state.extractor_errors,
                )

            try:
                content, state_regular_links, state_navigation_links = await self._build_html_document_output(
                    raw_html=state.html,
                    base_url=canonical_source,
                    depth=depth,
                    backend=backend,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                self._record_error(state_source, depth, "parse", str(exc), backend)
                if not self.continue_on_error:
                    raise
                continue

            if state.extra_links:
                state_regular_links, state_navigation_links = partition_web_links(
                    merge_web_links(state_regular_links, state_navigation_links, state.extra_links)
                )

            combined_regular_links.extend(state_regular_links)
            combined_navigation_links.extend(state_navigation_links)

            if mode == "single_document":
                if state_count > 1:
                    combined_content_parts.append(
                        f"\n\n===== NAVIGATION STATE {index + 1}/{state_count} =====\n\n{content}"
                    )
                else:
                    combined_content_parts.append(content)
                continue

            metadata: Dict[str, Any] = {
                "source": state_source,
                "source_type": "web",
                "output_format": self.output_format,
                "web_depth": depth,
                "parent_url": parent_url,
                "fetch_backend": backend,
                "start_url": self.url,
                "web_navigation_state_index": index,
                "web_navigation_state_count": state_count,
                "web_navigation_click_count": state.click_count,
                "web_navigation_state_hash": state.content_hash,
            }
            if state_count > 1:
                metadata["canonical_source"] = canonical_source
            if status_code is not None:
                metadata["status_code"] = status_code
            documents.append(Document(page_content=content, metadata=metadata))

        if mode == "single_document" and combined_content_parts:
            metadata: Dict[str, Any] = {
                "source": canonical_source,
                "source_type": "web",
                "output_format": self.output_format,
                "web_depth": depth,
                "parent_url": parent_url,
                "fetch_backend": backend,
                "start_url": self.url,
                "web_navigation_state_count": state_count,
                "web_navigation_click_count": max((state.click_count for state in states), default=0),
            }
            if status_code is not None:
                metadata["status_code"] = status_code
            documents.append(Document(page_content="".join(combined_content_parts), metadata=metadata))

        if not documents and first_error is not None:
            raise first_error

        merged_regular, merged_navigation = partition_web_links(
            merge_web_links(combined_regular_links, combined_navigation_links)
        )
        return documents, merged_regular, merged_navigation

    def _state_source_url(
        self,
        *,
        canonical_source: str,
        state_index: int,
        state_count: int,
        mode: NavigationStateDocumentMode,
    ) -> str:
        if mode != "separate_documents":
            return canonical_source
        if state_count <= 1:
            return canonical_source
        return f"{canonical_source}#nav-state={state_index}"

    async def _build_html_document_output(
        self,
        *,
        raw_html: str,
        base_url: str,
        depth: int,
        backend: str,
    ) -> tuple[str, list[WebLink], list[WebLink]]:
        document = parse_web_html_document(raw_html)
        regular_links, navigation_links = cleanup_and_extract_web_links(
            document,
            base_url=base_url,
            cleanup_config=self.cleanup_config,
            duplicate_hashes=self._duplicate_hashes,
            navigation_hashes=self._navigation_hashes,
        )
        custom_links = await self._run_custom_link_extractors(
            document=document,
            base_url=base_url,
            depth=depth,
            backend=backend,
        )
        if custom_links:
            merged = merge_web_links(regular_links, navigation_links, custom_links)
            regular_links, navigation_links = partition_web_links(merged)
        content = render_web_html_document(document, output_format=self.output_format)
        return content, regular_links, navigation_links

    async def _try_playwright_parse_fallback(
        self,
        *,
        url: str,
        primary_backend: str,
    ) -> tuple[_FetchResult | None, str | None]:
        if self.fetch_mode != "requests_fallback_playwright" or primary_backend != "requests":
            return None, None

        try:
            fallback_result = await self._fetch_via_playwright_async(url)
        except Exception as exc:
            return None, f"Playwright fallback unavailable: {exc}"

        if fallback_result.error:
            return None, f"Playwright fallback fetch failed: {fallback_result.error}"

        fallback_url = normalize_url(fallback_result.final_url or url)
        if not is_html_response(fallback_url, fallback_result.content_type):
            return None, "Playwright fallback response is not HTML."

        has_html = bool(
            fallback_result.navigation_states
            or fallback_result.text
            or fallback_result.content_bytes
        )
        if not has_html:
            return None, "Playwright fallback HTML response is empty."

        return fallback_result, None

    async def _run_custom_link_extractors(
        self,
        *,
        document: Any,
        base_url: str,
        depth: int,
        backend: str,
    ) -> list[WebLink]:
        if not self.custom_link_extractors:
            return []

        custom_links: list[WebLink] = []
        for extractor in self.custom_link_extractors:
            try:
                extracted = extractor(document, base_url)
                if inspect.isawaitable(extracted):
                    extracted = await extracted
            except Exception as exc:
                self._record_error(base_url, depth, "parse", f"custom_link_extractor failed: {exc}", backend)
                continue

            if extracted is None:
                continue

            if isinstance(extracted, (str, WebLink, tuple)):
                extracted_items: Sequence[WebLinkInput] = [extracted]
            else:
                extracted_items = list(extracted)

            for item in extracted_items:
                try:
                    normalized = normalize_web_link_input(item, base_url=base_url)
                except Exception as exc:
                    self._record_error(base_url, depth, "parse", f"Invalid custom link item: {exc}", backend)
                    continue
                if normalized is not None:
                    custom_links.append(normalized)

        return merge_web_links(custom_links)

    def _is_non_recursive_link(self, link: WebLink) -> bool:
        if self.cleanup_config is None:
            return False
        rules = self.cleanup_config.non_recursive_classes
        if not rules:
            return False

        source_classes = set(link.source_classes)
        if not source_classes:
            return False

        for rule in rules:
            tokens = [part.strip() for part in str(rule).split() if part.strip()]
            if not tokens:
                continue
            if len(tokens) == 1 and tokens[0] in source_classes:
                return True
            if len(tokens) > 1 and all(token in source_classes for token in tokens):
                return True
        return False

    def _annotate_download_documents(
        self,
        *,
        docs: Sequence[Document],
        depth: int,
        parent_url: str | None,
        backend: str,
    ) -> list[Document]:
        wrapped: list[Document] = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata["web_depth"] = depth
            metadata["parent_url"] = parent_url
            metadata["start_url"] = self.url
            metadata["fetch_backend"] = backend
            wrapped.append(Document(page_content=doc.page_content, metadata=metadata))
        return wrapped

    async def _fetch_url(self, url: str) -> _FetchResult:
        if self.fetch_mode == "requests":
            return await self._fetch_via_requests_async(url)

        if self.fetch_mode == "playwright":
            return await self._fetch_via_playwright_async(url)

        request_result = await self._fetch_via_requests_async(url)
        should_fallback = bool(request_result.error) or request_result.status_code in {401, 403}
        if should_fallback:
            playwright_result = await self._fetch_via_playwright_async(url)
            if not playwright_result.error:
                return playwright_result
            if request_result.error:
                return playwright_result
        return request_result

    async def _fetch_via_requests_async(self, url: str) -> _FetchResult:
        last_error = ""
        for _attempt in range(self.retry_attempts + 1):
            try:
                response = await asyncio.to_thread(self._requests_get, url)
                content_type = normalize_content_type(response.headers.get("Content-Type", ""))
                final_url = normalize_url(response.url or url)
                text = ""
                if is_html_response(final_url, content_type):
                    text = response.text or ""
                return _FetchResult(
                    backend="requests",
                    url=url,
                    final_url=final_url,
                    status_code=response.status_code,
                    content_type=content_type,
                    headers={str(k): str(v) for k, v in response.headers.items()},
                    text=text,
                    content_bytes=response.content or b"",
                )
            except Exception as exc:
                last_error = str(exc)
        return _FetchResult(backend="requests", url=url, final_url=url, error=last_error)

    def _requests_get(self, url: str) -> Any:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required for AsyncWebLoader requests modes.") from exc
        return requests.get(
            url,
            timeout=self.request_timeout_seconds,
            allow_redirects=True,
            headers={"User-Agent": self.user_agent},
            verify=not self.ignore_https_errors,
        )

    async def _ensure_playwright_context(self) -> None:
        if self._playwright_context is not None:
            return

        async with self._playwright_init_lock:
            if self._playwright_context is not None:
                return
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise ImportError(
                    "playwright is required for fetch_mode='playwright' or fallback mode. "
                    "Install with `pip install rag-lib[web]` and `playwright install chromium`."
                ) from exc

            self._playwright_driver = await async_playwright().start()
            self._playwright_browser = await self._playwright_driver.chromium.launch(
                headless=self.playwright_headless
            )
            self._playwright_context = await self._playwright_browser.new_context(
                user_agent=self.user_agent,
                accept_downloads=True,
                ignore_https_errors=self.ignore_https_errors,
            )

    async def _fetch_via_playwright_async(self, url: str) -> _FetchResult:
        await self._ensure_playwright_context()
        assert self._playwright_context is not None

        last_error = ""
        for _attempt in range(self.retry_attempts + 1):
            page = await self._playwright_context.new_page()
            try:
                response = await page.goto(url, timeout=self.playwright_timeout_ms, wait_until="load")
                status_code = response.status if response is not None else None
                headers = dict(response.headers) if response is not None else {}
                final_url = normalize_url(page.url or url)

                if self._should_trigger_login(status_code=status_code, final_url=final_url):
                    if not await self._run_login_processor(page=page, current_url=final_url):
                        return _FetchResult(
                            backend="playwright",
                            url=url,
                            final_url=final_url,
                            status_code=status_code,
                            content_type=normalize_content_type(get_header(headers, "content-type")),
                            headers=headers,
                            error="Authentication required and login_processor failed or not provided.",
                        )
                    response = await page.goto(url, timeout=self.playwright_timeout_ms, wait_until="load")
                    status_code = response.status if response is not None else None
                    headers = dict(response.headers) if response is not None else {}
                    final_url = normalize_url(page.url or url)

                content_type = normalize_content_type(get_header(headers, "content-type"))
                body_bytes = b""
                if response is not None:
                    try:
                        body_bytes = await response.body()
                    except Exception:
                        body_bytes = b""

                text = ""
                extra_links: list[WebLink] = []
                extractor_errors: list[str] = []
                navigation_states: list[PlaywrightNavigationState] = []
                navigation_click_count = 0
                if is_html_response(final_url, content_type):
                    extra_links, extractor_errors = await self._run_playwright_link_extractor(
                        page=page,
                        base_url=final_url,
                    )
                    text = await page.content() or ""
                    if not body_bytes and text:
                        body_bytes = text.encode("utf-8")

                    navigation_states.append(
                        PlaywrightNavigationState(
                            html=text,
                            content_hash=hashlib.md5(text.encode("utf-8")).hexdigest(),
                            click_count=0,
                            extra_links=tuple(extra_links),
                            extractor_errors=tuple(extractor_errors),
                        )
                    )

                    if self._should_run_playwright_navigation():
                        try:
                            navigation_result = await run_async_cleanup_navigation(
                                page=page,
                                base_url=final_url,
                                cleanup_config=self.cleanup_config,
                                config=self.playwright_navigation_config,
                                on_state=lambda page_obj, base: self._run_playwright_link_extractor(
                                    page=page_obj,
                                    base_url=base,
                                ),
                            )
                        except Exception as exc:
                            extractor_errors.append(f"cleanup_navigation: {exc}")
                        else:
                            navigation_click_count = navigation_result.click_count
                            if navigation_result.states:
                                navigation_states.extend(list(navigation_result.states))
                            if navigation_result.errors:
                                extractor_errors.extend(
                                    [f"cleanup_navigation: {message}" for message in navigation_result.errors]
                                )

                return _FetchResult(
                    backend="playwright",
                    url=url,
                    final_url=final_url,
                    status_code=status_code,
                    content_type=content_type,
                    headers=headers,
                    text=text,
                    content_bytes=body_bytes,
                    extra_links=extra_links,
                    extractor_errors=extractor_errors,
                    navigation_states=navigation_states,
                    navigation_click_count=navigation_click_count,
                )
            except Exception as exc:
                last_error = str(exc)
            finally:
                await page.close()

        return _FetchResult(backend="playwright", url=url, final_url=url, error=last_error)

    async def _run_playwright_link_extractor(self, *, page: Any, base_url: str) -> tuple[list[WebLink], list[str]]:
        links: list[WebLink] = []
        errors: list[str] = []

        if self.playwright_extraction_config is not None:
            profile_links, profile_errors = await run_async_playwright_extraction(
                config=self.playwright_extraction_config,
                page=page,
                base_url=base_url,
            )
            links.extend(profile_links)
            errors.extend(profile_errors)

        if self.playwright_link_extractor is not None:
            try:
                extracted = self.playwright_link_extractor(page, base_url)
                if inspect.isawaitable(extracted):
                    extracted = await extracted
            except Exception as exc:
                errors.append(f"legacy_extractor: {exc}")
            else:
                if extracted is not None:
                    if isinstance(extracted, (str, WebLink, tuple)):
                        extracted_items: Sequence[WebLinkInput] = [extracted]
                    else:
                        extracted_items = list(extracted)

                    for item in extracted_items:
                        try:
                            normalized = normalize_web_link_input(item, base_url=base_url)
                        except Exception as exc:
                            errors.append(f"legacy_extractor_item: {exc}")
                            continue
                        if normalized is not None:
                            links.append(normalized)

        return merge_web_links(links), errors

    def _should_run_playwright_navigation(self) -> bool:
        if self.cleanup_config is None:
            return False
        return any(
            (
                bool(self.cleanup_config.navigation_classes),
                bool(self.cleanup_config.navigation_styles),
                bool(self.cleanup_config.navigation_texts),
            )
        ) and self.playwright_navigation_config.enabled

    def _record_playwright_extractor_errors(
        self,
        *,
        url: str,
        depth: int,
        backend: str,
        messages: Sequence[str],
    ) -> None:
        for message in messages:
            self._record_error(
                url,
                depth,
                "parse",
                f"playwright_link_extractor: {message}",
                backend,
            )

    def _should_trigger_login(self, *, status_code: Optional[int], final_url: str) -> bool:
        if status_code in {401, 403}:
            return True

        normalized_final = normalize_url(final_url).lower()
        if self.login_url and normalized_final.startswith(self.login_url.lower()):
            return True

        parsed = urlparse(normalized_final)
        path_segments = [segment for segment in (parsed.path or "").split("/") if segment]
        login_markers = {"login", "signin", "sign-in", "auth"}
        return any(segment in login_markers for segment in path_segments)

    async def _run_login_processor(self, *, page: Any, current_url: str) -> bool:
        if self.login_processor is None:
            return False
        if self._playwright_context is None:
            return False
        if self._login_completed:
            return True

        async with self._login_lock:
            if self._login_completed:
                return True
            try:
                result = self.login_processor(
                    page,
                    self._playwright_context,
                    self.url,
                    self.login_url,
                    current_url,
                )
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                self._record_error(current_url, 0, "auth", str(exc), "playwright")
                return False

            if result is False:
                return False
            self._login_completed = True
            return True

    def _record_error(self, url: str, depth: int, stage: str, error: str, backend: str) -> None:
        self.last_errors.append(
            {
                "url": url,
                "depth": depth,
                "stage": stage,
                "error": error,
                "backend": backend,
            }
        )

    def _record_filter(self, url: str, depth: int, message: str, backend: str) -> None:
        self._record_error(url, depth, "filter", message, backend)

    async def _close_resources(self) -> None:
        if self._playwright_context is not None:
            await self._playwright_context.close()
            self._playwright_context = None

        if self._playwright_browser is not None:
            await self._playwright_browser.close()
            self._playwright_browser = None

        if self._playwright_driver is not None:
            await self._playwright_driver.stop()
            self._playwright_driver = None
