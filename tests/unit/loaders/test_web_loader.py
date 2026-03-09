from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from rag_lib.core.domain import Document
from rag_lib.loaders import web_common
from rag_lib.loaders.web import WebLoader, _FetchResult as SyncFetchResult
from rag_lib.loaders.web_async import AsyncWebLoader, _FetchResult as AsyncFetchResult
from rag_lib.loaders.web_common import WebCleanupConfig, WebLink
from rag_lib.loaders.web_playwright_extractors import (
    PlaywrightExtractionConfig,
    PlaywrightNavigationConfig,
    PlaywrightNavigationState,
    PlaywrightProfileConfig,
)


def _sync_html_result(url: str, html: str) -> SyncFetchResult:
    return SyncFetchResult(
        backend="requests",
        url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        headers={"Content-Type": "text/html"},
        text=html,
        content_bytes=html.encode("utf-8"),
    )


def _sync_download_result(
    url: str,
    content_type: str,
    payload: bytes,
    headers: dict[str, str] | None = None,
) -> SyncFetchResult:
    return SyncFetchResult(
        backend="requests",
        url=url,
        final_url=url,
        status_code=200,
        content_type=content_type,
        headers=headers or {"Content-Type": content_type},
        content_bytes=payload,
    )


def _async_html_result(url: str, html: str, backend: str = "requests") -> AsyncFetchResult:
    return AsyncFetchResult(
        backend=backend,
        url=url,
        final_url=url,
        status_code=200,
        content_type="text/html",
        headers={"Content-Type": "text/html"},
        text=html,
        content_bytes=html.encode("utf-8"),
    )


def _async_download_result(
    url: str,
    content_type: str,
    payload: bytes,
    *,
    headers: dict[str, str] | None = None,
    backend: str = "requests",
) -> AsyncFetchResult:
    return AsyncFetchResult(
        backend=backend,
        url=url,
        final_url=url,
        status_code=200,
        content_type=content_type,
        headers=headers or {"Content-Type": content_type},
        content_bytes=payload,
    )


def test_sync_web_loader_download_flag_defaults_to_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        follow_download_links=False,
    )

    def fake_fetch(url: str) -> SyncFetchResult:
        if url == "https://example.com/root":
            return _sync_html_result(url, '<html><body><a href="/file.pdf">PDF</a></body></html>')
        return _sync_download_result(url, "application/pdf", b"%PDF-1.4")

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "web"
    assert loader.last_stats["skipped_count"] >= 1


def test_sync_web_loader_tolerates_empty_list_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests",
    )

    def fake_fetch(url: str) -> SyncFetchResult:
        assert url == "https://example.com/root"
        return _sync_html_result(
            url,
            (
                "<html><body>"
                "<ul><li> </li><li>&nbsp;</li></ul>"
                "<p>kept</p>"
                "</body></html>"
            ),
        )

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    assert "kept" in docs[0].page_content
    assert loader.last_stats["error_count"] == 0


def test_sync_web_loader_handles_vue_style_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests",
    )

    def fake_fetch(url: str) -> SyncFetchResult:
        assert url == "https://example.com/root"
        return _sync_html_result(
            url,
            (
                "<html><body>"
                "<div :class=\"'x'\" @click=\"do()\">"
                "<img :src=\"imgUrl\" />"
                "<p>kept</p>"
                "</div>"
                "</body></html>"
            ),
        )

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    assert "kept" in docs[0].page_content
    assert loader.last_stats["error_count"] == 0


def test_sync_navigation_same_depth_and_non_recursive_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        crawl_scope="same_host",
        cleanup_config=WebCleanupConfig(
            navigation_classes=("pager",),
            non_recursive_classes=("tag",),
        ),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<div class='pager'><a href='/nav1' class='tag'>nav1</a></div>"
            "<a href='/regular1' class='tag'>skip</a>"
            "<a href='/regular2'>ok</a>"
            "</body></html>"
        ),
        "https://example.com/nav1": "<html><body><div class='pager'><a href='/nav2'>nav2</a></div></body></html>",
        "https://example.com/nav2": "<html><body><p>nav2 page</p></body></html>",
        "https://example.com/regular2": "<html><body><p>regular2 page</p></body></html>",
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/nav1" in sources
    assert "https://example.com/nav2" in sources
    assert "https://example.com/regular2" in sources
    assert "https://example.com/regular1" not in sources
    assert loader.last_stats["visited_count"] == 4
    assert loader.last_stats["max_depth_reached"] == 1


def test_sync_cleanup_ignored_classes_remove_content_and_links(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(ignored_classes=("footer",)),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<div class='footer'><a href='/hidden'>hidden</a><p>ignored</p></div>"
            "<p>kept</p>"
            "</body></html>"
        )
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    assert "kept" in docs[0].page_content
    assert "ignored" not in docs[0].page_content
    assert loader.last_stats["visited_count"] == 1


def test_sync_navigation_style_rules_extract_same_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(
            navigation_styles=("display: table-cell;",),
        ),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<div style='display: table-cell;'><a href='/nav'>nav</a></div>"
            "<a href='/child'>child</a>"
            "</body></html>"
        ),
        "https://example.com/nav": "<html><body><p>nav page</p></body></html>",
        "https://example.com/child": "<html><body><p>child page</p></body></html>",
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/nav" in sources
    assert "https://example.com/child" in sources
    assert loader.last_stats["visited_count"] == 3


def test_sync_navigation_style_rules_do_not_strip_rendered_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(
            navigation_styles=("display: table-cell;",),
        ),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<div style='display: table-cell;'><a href='/nav'>nav</a><p>kept style block</p></div>"
            "</body></html>"
        ),
        "https://example.com/nav": "<html><body><p>nav page</p></body></html>",
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)
    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    root_doc = next(doc for doc in docs if doc.metadata["source"] == "https://example.com/root")
    assert "kept style block" in root_doc.page_content


def test_sync_navigation_text_markers_are_removed_from_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        output_format="html",
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(navigation_texts=("<", ">")),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<table><thead><tr><th style='display: table-cell;'><a>&lt;</a> <a>&gt;</a></th></tr></thead></table>"
            "<p>kept</p>"
            "</body></html>"
        )
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    assert "kept" in docs[0].page_content
    assert "&lt;" not in docs[0].page_content
    assert "&gt;" not in docs[0].page_content


def test_sync_duplicate_tags_deduplicates_across_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(duplicate_tags=("div",)),
    )

    pages = {
        "https://example.com/root": "<html><body><div>same-block</div><a href='/child'>child</a></body></html>",
        "https://example.com/child": "<html><body><div>same-block</div><p>child-only</p></body></html>",
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    root_doc = next(doc for doc in docs if doc.metadata["source"].endswith("/root"))
    child_doc = next(doc for doc in docs if doc.metadata["source"].endswith("/child"))
    assert "same-block" in root_doc.page_content
    assert "child-only" in child_doc.page_content
    assert "same-block" not in child_doc.page_content


def test_sync_custom_link_extractors_support_url_only_and_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(non_recursive_classes=("tag",)),
        custom_link_extractors=[
            lambda _document, _url: [
                "/custom1",
                WebLink(url="/navcustom", is_navigation=True),
                ("/custom2", ["tag"]),
            ]
        ],
    )

    pages = {
        "https://example.com/root": "<html><body><p>root</p></body></html>",
        "https://example.com/custom1": "<html><body><p>custom1</p></body></html>",
        "https://example.com/navcustom": "<html><body><p>nav custom</p></body></html>",
    }

    def fake_fetch(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _sync_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/custom1" in sources
    assert "https://example.com/navcustom" in sources
    assert "https://example.com/custom2" not in sources


def test_sync_filter_stage_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        cleanup_config=WebCleanupConfig(non_recursive_classes=("tag",)),
    )

    def fake_fetch(url: str) -> SyncFetchResult:
        if url == "https://example.com/root":
            return _sync_html_result(
                url,
                (
                    "<html><body>"
                    "<a href='/skip' class='tag'>skip</a>"
                    "<a href='https://outside.example/x'>outside</a>"
                    "</body></html>"
                ),
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_fetch)

    docs = loader.load()

    assert len(docs) == 1
    filter_entries = [err for err in loader.last_errors if err["stage"] == "filter"]
    assert len(filter_entries) >= 2
    assert loader.last_stats["error_count"] == 0


def test_sync_playwright_extra_links_are_crawled(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="playwright",
    )

    pages = {
        "https://example.com/root": (
            "<html><body><p>root</p></body></html>",
            [
                WebLink(url="https://example.com/nav", is_navigation=True),
                WebLink(url="https://example.com/child"),
            ],
        ),
        "https://example.com/nav": ("<html><body><p>nav</p></body></html>", []),
        "https://example.com/child": ("<html><body><p>child</p></body></html>", []),
    }

    def fake_playwright(url: str) -> SyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        html, extra_links = pages[url]
        return SyncFetchResult(
            backend="playwright",
            url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=html,
            content_bytes=html.encode("utf-8"),
            extra_links=extra_links,
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)

    docs = loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert sources == {
        "https://example.com/root",
        "https://example.com/nav",
        "https://example.com/child",
    }


def test_sync_playwright_extraction_config_combines_with_legacy_extractor() -> None:
    class _Page:
        def evaluate(self, _script, _arg=None):
            return ["/from-config", "/shared"]

        def wait_for_timeout(self, _timeout_ms):
            return None

    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
        playwright_extraction_config=PlaywrightExtractionConfig(
            profiles=(PlaywrightProfileConfig(profile="eval", script="() => []"),)
        ),
        playwright_link_extractor=lambda _page, _url: ["/legacy", "/shared"],
    )

    links, errors = loader._run_playwright_link_extractor(
        page=_Page(),
        base_url="https://example.com/root",
    )

    urls = {link.url for link in links}
    assert urls == {
        "https://example.com/from-config",
        "https://example.com/shared",
        "https://example.com/legacy",
    }
    assert errors == []


def test_sync_playwright_extractor_errors_are_recorded_and_crawl_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
    )

    def fake_playwright(_url: str) -> SyncFetchResult:
        html = "<html><body><p>ok</p></body></html>"
        return SyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=html,
            content_bytes=html.encode("utf-8"),
            extractor_errors=["eval: failed to execute"],
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)

    docs = loader.load()

    assert len(docs) == 1
    assert any(
        err["stage"] == "parse" and "playwright_link_extractor" in err["error"]
        for err in loader.last_errors
    )
    assert loader.last_stats["error_count"] >= 1


def test_sync_cleanup_navigation_errors_are_recorded_and_crawl_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
    )

    def fake_playwright(_url: str) -> SyncFetchResult:
        html = "<html><body><p>ok</p></body></html>"
        return SyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=html,
            content_bytes=html.encode("utf-8"),
            extractor_errors=["cleanup_navigation: click failed"],
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)
    docs = loader.load()

    assert len(docs) == 1
    assert any("cleanup_navigation" in err["error"] for err in loader.last_errors)


def test_sync_playwright_navigation_states_emit_separate_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
    )

    states = [
        PlaywrightNavigationState(
            html="<html><body><p>state-0</p></body></html>",
            content_hash="h0",
            click_count=0,
        ),
        PlaywrightNavigationState(
            html="<html><body><p>state-1</p></body></html>",
            content_hash="h1",
            click_count=1,
        ),
        PlaywrightNavigationState(
            html="<html><body><p>state-2</p></body></html>",
            content_hash="h2",
            click_count=2,
        ),
    ]

    def fake_playwright(_url: str) -> SyncFetchResult:
        return SyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=states[0].html,
            content_bytes=states[0].html.encode("utf-8"),
            navigation_states=states,
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)
    docs = loader.load()

    assert len(docs) == 3
    assert docs[0].metadata["source"].endswith("#nav-state=0")
    assert docs[1].metadata["source"].endswith("#nav-state=1")
    assert docs[2].metadata["source"].endswith("#nav-state=2")
    assert docs[0].metadata["canonical_source"] == "https://example.com/root"
    assert docs[2].metadata["web_navigation_state_count"] == 3


def test_sync_links_from_later_navigation_states_are_crawled(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="playwright",
    )

    root_states = [
        PlaywrightNavigationState(
            html="<html><body><p>state-0</p></body></html>",
            content_hash="h0",
            click_count=0,
        ),
        PlaywrightNavigationState(
            html="<html><body><a href='/from-state-1'>s1</a></body></html>",
            content_hash="h1",
            click_count=1,
        ),
        PlaywrightNavigationState(
            html="<html><body><a href='/from-state-2'>s2</a></body></html>",
            content_hash="h2",
            click_count=2,
        ),
    ]

    def fake_playwright(url: str) -> SyncFetchResult:
        if url == "https://example.com/root":
            return SyncFetchResult(
                backend="playwright",
                url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                headers={"Content-Type": "text/html"},
                text=root_states[0].html,
                content_bytes=root_states[0].html.encode("utf-8"),
                navigation_states=root_states,
            )
        if url == "https://example.com/from-state-1":
            return _sync_html_result(url, "<html><body><p>c1</p></body></html>")
        if url == "https://example.com/from-state-2":
            return _sync_html_result(url, "<html><body><p>c2</p></body></html>")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)
    docs = loader.load()
    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/from-state-1" in sources
    assert "https://example.com/from-state-2" in sources


def test_sync_playwright_navigation_disabled_prevents_auto_run() -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
        cleanup_config=WebCleanupConfig(navigation_texts=(">",)),
        playwright_navigation_config=PlaywrightNavigationConfig(enabled=False),
    )
    assert loader._should_run_playwright_navigation() is False


@pytest.mark.asyncio
async def test_async_web_loader_depth_and_scope_with_mocked_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        crawl_scope="same_host",
        max_concurrency=3,
    )

    async def fake_fetch(url: str) -> AsyncFetchResult:
        if url == "https://example.com/root":
            return _async_html_result(
                url,
                (
                    '<html><body>'
                    '<a href="/child">child</a>'
                    '<a href="https://external.test/page">external</a>'
                    "</body></html>"
                ),
            )
        if url == "https://example.com/child":
            return _async_html_result(url, "<html><body><p>child</p></body></html>")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_fetch)

    docs = await loader.load()

    assert len(docs) == 2
    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/child" in sources
    assert loader.last_stats["visited_count"] == 2
    assert loader.last_stats["error_count"] == 0


@pytest.mark.asyncio
async def test_async_playwright_extra_links_are_crawled(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="playwright",
        max_concurrency=2,
    )

    pages = {
        "https://example.com/root": (
            "<html><body><p>root</p></body></html>",
            [
                WebLink(url="https://example.com/nav", is_navigation=True),
                WebLink(url="https://example.com/child"),
            ],
        ),
        "https://example.com/nav": ("<html><body><p>nav</p></body></html>", []),
        "https://example.com/child": ("<html><body><p>child</p></body></html>", []),
    }

    async def fake_playwright(url: str) -> AsyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        html, extra_links = pages[url]
        return AsyncFetchResult(
            backend="playwright",
            url=url,
            final_url=url,
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=html,
            content_bytes=html.encode("utf-8"),
            extra_links=extra_links,
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)

    docs = await loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert sources == {
        "https://example.com/root",
        "https://example.com/nav",
        "https://example.com/child",
    }


@pytest.mark.asyncio
async def test_async_playwright_extractor_errors_are_recorded_and_crawl_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
    )

    async def fake_playwright(_url: str) -> AsyncFetchResult:
        html = "<html><body><p>ok</p></body></html>"
        return AsyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=html,
            content_bytes=html.encode("utf-8"),
            extractor_errors=["eval: async failed"],
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)

    docs = await loader.load()

    assert len(docs) == 1
    assert any(
        err["stage"] == "parse" and "playwright_link_extractor" in err["error"]
        for err in loader.last_errors
    )
    assert loader.last_stats["error_count"] >= 1


@pytest.mark.asyncio
async def test_async_playwright_navigation_states_emit_separate_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
    )

    states = [
        PlaywrightNavigationState(
            html="<html><body><p>state-0</p></body></html>",
            content_hash="h0",
            click_count=0,
        ),
        PlaywrightNavigationState(
            html="<html><body><p>state-1</p></body></html>",
            content_hash="h1",
            click_count=1,
        ),
        PlaywrightNavigationState(
            html="<html><body><p>state-2</p></body></html>",
            content_hash="h2",
            click_count=2,
        ),
    ]

    async def fake_playwright(_url: str) -> AsyncFetchResult:
        return AsyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text=states[0].html,
            content_bytes=states[0].html.encode("utf-8"),
            navigation_states=states,
        )

    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)
    docs = await loader.load()

    assert len(docs) == 3
    assert docs[0].metadata["source"].endswith("#nav-state=0")
    assert docs[1].metadata["source"].endswith("#nav-state=1")
    assert docs[2].metadata["source"].endswith("#nav-state=2")
    assert docs[0].metadata["canonical_source"] == "https://example.com/root"
    assert docs[2].metadata["web_navigation_state_count"] == 3


@pytest.mark.asyncio
async def test_async_links_from_later_navigation_states_are_crawled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="playwright",
    )

    root_states = [
        PlaywrightNavigationState(
            html="<html><body><p>state-0</p></body></html>",
            content_hash="h0",
            click_count=0,
        ),
        PlaywrightNavigationState(
            html="<html><body><a href='/from-state-1'>s1</a></body></html>",
            content_hash="h1",
            click_count=1,
        ),
        PlaywrightNavigationState(
            html="<html><body><a href='/from-state-2'>s2</a></body></html>",
            content_hash="h2",
            click_count=2,
        ),
    ]

    async def fake_playwright(url: str) -> AsyncFetchResult:
        if url == "https://example.com/root":
            return AsyncFetchResult(
                backend="playwright",
                url=url,
                final_url=url,
                status_code=200,
                content_type="text/html",
                headers={"Content-Type": "text/html"},
                text=root_states[0].html,
                content_bytes=root_states[0].html.encode("utf-8"),
                navigation_states=root_states,
            )
        if url == "https://example.com/from-state-1":
            return _async_html_result(url, "<html><body><p>c1</p></body></html>", backend="playwright")
        if url == "https://example.com/from-state-2":
            return _async_html_result(url, "<html><body><p>c2</p></body></html>", backend="playwright")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)
    docs = await loader.load()
    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/from-state-1" in sources
    assert "https://example.com/from-state-2" in sources


@pytest.mark.asyncio
async def test_async_playwright_navigation_disabled_prevents_auto_run() -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
        cleanup_config=WebCleanupConfig(navigation_texts=(">",)),
        playwright_navigation_config=PlaywrightNavigationConfig(enabled=False),
    )
    assert loader._should_run_playwright_navigation() is False


@pytest.mark.asyncio
async def test_async_navigation_same_depth_and_non_recursive_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        crawl_scope="same_host",
        max_concurrency=4,
        cleanup_config=WebCleanupConfig(
            navigation_classes=("pager",),
            non_recursive_classes=("tag",),
        ),
    )

    pages = {
        "https://example.com/root": (
            "<html><body>"
            "<div class='pager'><a href='/nav1' class='tag'>nav1</a></div>"
            "<a href='/regular1' class='tag'>skip</a>"
            "<a href='/regular2'>ok</a>"
            "</body></html>"
        ),
        "https://example.com/nav1": "<html><body><div class='pager'><a href='/nav2'>nav2</a></div></body></html>",
        "https://example.com/nav2": "<html><body><p>nav2 page</p></body></html>",
        "https://example.com/regular2": "<html><body><p>regular2 page</p></body></html>",
    }

    async def fake_fetch(url: str) -> AsyncFetchResult:
        if url not in pages:
            raise AssertionError(f"Unexpected URL: {url}")
        return _async_html_result(url, pages[url])

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_fetch)

    docs = await loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/nav1" in sources
    assert "https://example.com/nav2" in sources
    assert "https://example.com/regular2" in sources
    assert "https://example.com/regular1" not in sources
    assert loader.last_stats["visited_count"] == 4
    assert loader.last_stats["max_depth_reached"] == 1


@pytest.mark.asyncio
async def test_async_web_loader_fallback_to_playwright_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests_fallback_playwright",
    )

    async def fake_requests(_url: str) -> AsyncFetchResult:
        return AsyncFetchResult(
            backend="requests",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=401,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
        )

    fake_playwright = AsyncMock(
        return_value=_async_html_result(
            "https://example.com/root",
            "<html><body><p>ok</p></body></html>",
            backend="playwright",
        )
    )

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_requests)
    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)

    docs = await loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["fetch_backend"] == "playwright"
    assert fake_playwright.await_count == 1


def test_sync_web_loader_fallback_to_playwright_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests_fallback_playwright",
    )

    def fake_requests(_url: str) -> SyncFetchResult:
        return _sync_html_result("https://example.com/root", "<html><body><p>bad</p></body></html>")

    def fake_playwright(_url: str) -> SyncFetchResult:
        return SyncFetchResult(
            backend="playwright",
            url="https://example.com/root",
            final_url="https://example.com/root",
            status_code=200,
            content_type="text/html",
            headers={"Content-Type": "text/html"},
            text="<html><body><p>ok</p></body></html>",
            content_bytes=b"<html><body><p>ok</p></body></html>",
        )

    original_build = loader._build_html_document_output

    def fake_build(*, raw_html: str, base_url: str, depth: int, backend: str):
        if backend == "requests":
            raise ValueError("parse-failed")
        return original_build(raw_html=raw_html, base_url=base_url, depth=depth, backend=backend)

    monkeypatch.setattr(loader, "_fetch_via_requests", fake_requests)
    monkeypatch.setattr(loader, "_fetch_via_playwright", fake_playwright)
    monkeypatch.setattr(loader, "_build_html_document_output", fake_build)

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["fetch_backend"] == "playwright"
    assert "ok" in docs[0].page_content


@pytest.mark.asyncio
async def test_async_web_loader_fallback_to_playwright_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests_fallback_playwright",
    )

    async def fake_requests(_url: str) -> AsyncFetchResult:
        return _async_html_result(
            "https://example.com/root",
            "<html><body><p>bad</p></body></html>",
            backend="requests",
        )

    async def fake_playwright(_url: str) -> AsyncFetchResult:
        return _async_html_result(
            "https://example.com/root",
            "<html><body><p>ok</p></body></html>",
            backend="playwright",
        )

    original_build = loader._build_html_document_output

    async def fake_build(*, raw_html: str, base_url: str, depth: int, backend: str):
        if backend == "requests":
            raise ValueError("parse-failed")
        return await original_build(raw_html=raw_html, base_url=base_url, depth=depth, backend=backend)

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_requests)
    monkeypatch.setattr(loader, "_fetch_via_playwright_async", fake_playwright)
    monkeypatch.setattr(loader, "_build_html_document_output", fake_build)

    docs = await loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["fetch_backend"] == "playwright"
    assert "ok" in docs[0].page_content


@pytest.mark.asyncio
async def test_async_login_trigger_and_callback_contract() -> None:
    called: list[tuple[object, object, str, str | None, str]] = []

    async def login_cb(page, context, start_url, login_url, current_url):
        called.append((page, context, start_url, login_url, current_url))
        return True

    loader = AsyncWebLoader(
        url="https://example.com/root",
        fetch_mode="playwright",
        login_url="https://example.com/login",
        login_processor=login_cb,
    )
    loader._playwright_context = object()

    assert loader._should_trigger_login(status_code=401, final_url="https://example.com/root")
    assert loader._should_trigger_login(status_code=403, final_url="https://example.com/root")
    assert loader._should_trigger_login(status_code=200, final_url="https://example.com/login")

    page = object()
    ok = await loader._run_login_processor(page=page, current_url="https://example.com/login")
    assert ok is True
    assert loader._login_completed is True
    assert len(called) == 1
    assert called[0][0] is page
    assert called[0][1] is loader._playwright_context
    assert called[0][2] == "https://example.com/root"
    assert called[0][3] == "https://example.com/login"
    assert called[0][4] == "https://example.com/login"


def test_download_detection_by_header_and_extension() -> None:
    assert web_common.is_download_response(
        "https://example.com/file",
        "application/octet-stream",
        {"Content-Disposition": 'attachment; filename="report.pdf"'},
    )
    assert (
        web_common.infer_download_kind(
            "https://example.com/file",
            "application/octet-stream",
            {"Content-Disposition": 'attachment; filename="report.pdf"'},
        )
        == "pdf"
    )

    assert (
        web_common.infer_download_kind(
            "https://example.com/data.csv",
            "",
            {},
        )
        == "csv"
    )


def test_sync_playwright_visible_alias_sets_headful_mode() -> None:
    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
        playwright_visible=True,
    )
    assert loader.playwright_headless is False


def test_async_playwright_visible_alias_sets_headful_mode() -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="playwright",
        playwright_visible=True,
    )
    assert loader.playwright_headless is False


def test_sync_loader_ignore_https_errors_disables_requests_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.verify: bool = True

        def close(self) -> None:
            return None

    class _FakeRequests:
        Session = _FakeSession

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests)

    loader = WebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests",
        ignore_https_errors=True,
    )
    session = loader._ensure_requests_session()
    assert session.verify is False


def test_async_loader_ignore_https_errors_disables_requests_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def _fake_get(_url: str, **kwargs):
        seen.update(kwargs)
        return object()

    class _FakeRequests:
        get = staticmethod(_fake_get)

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests)

    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=0,
        fetch_mode="requests",
        ignore_https_errors=True,
    )
    loader._requests_get("https://example.com/root")
    assert seen["verify"] is False


def test_download_routing_selects_supported_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return [Document(page_content="ok", metadata={"source": "tmp"})]

    monkeypatch.setattr(web_common, "PDFLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "PyMuPDFLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "DocXLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "PPTXLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "HTMLLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "CSVLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "ExcelLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "JsonLoader", _FakeLoader)
    monkeypatch.setattr(web_common, "TextLoader", _FakeLoader)

    cases = [
        ("https://example.com/a.pdf", "application/pdf", "PDFLoader"),
        (
            "https://example.com/a.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "DocXLoader",
        ),
        (
            "https://example.com/a.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "PPTXLoader",
        ),
        ("https://example.com/a.html", "text/html", "HTMLLoader"),
        ("https://example.com/a.csv", "text/csv", "CSVLoader"),
        (
            "https://example.com/a.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ExcelLoader",
        ),
        ("https://example.com/a.json", "application/json", "JsonLoader"),
        ("https://example.com/a.txt", "text/plain", "TextLoader"),
    ]

    for url, content_type, expected_loader in cases:
        docs = web_common.route_download_content_to_documents(
            content_bytes=b"dummy",
            source_url=url,
            content_type=content_type,
            headers={"Content-Type": content_type},
            output_format="markdown",
        )
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "web_download"
        assert docs[0].metadata["source"] == url
        assert docs[0].metadata["download_content_type"] == content_type
        assert docs[0].metadata["routed_loader"] == expected_loader


def test_download_routing_pdf_hook() -> None:
    seen: list[str] = []

    def hook(path: str) -> tuple[list[Document], str]:
        seen.append(path)
        return [Document(page_content="hook", metadata={"source": path})], "HookLoader"

    docs = web_common.route_download_content_to_documents(
        content_bytes=b"pdf-bytes",
        source_url="https://example.com/file.pdf",
        content_type="application/pdf",
        headers={"Content-Type": "application/pdf"},
        output_format="markdown",
        pdf_routing_hook=hook,
    )

    assert len(seen) == 1
    assert len(docs) == 1
    assert docs[0].metadata["routed_loader"] == "HookLoader"


@pytest.mark.asyncio
async def test_async_mixed_html_and_download_crawl(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        follow_download_links=True,
    )

    async def fake_fetch(url: str) -> AsyncFetchResult:
        if url == "https://example.com/root":
            return _async_html_result(
                url,
                (
                    '<html><body>'
                    '<a href="/child">child</a>'
                    '<a href="/report.pdf">report</a>'
                    "</body></html>"
                ),
            )
        if url == "https://example.com/child":
            return _async_html_result(url, "<html><body><p>child</p></body></html>")
        if url == "https://example.com/report.pdf":
            return _async_download_result(url, "application/pdf", b"%PDF-1.4")
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_route(**kwargs):
        return [
            Document(
                page_content="download-content",
                metadata={
                    "source": kwargs["source_url"],
                    "source_type": "web_download",
                    "routed_loader": "PDFLoader",
                },
            )
        ]

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_fetch)
    monkeypatch.setattr("rag_lib.loaders.web_async.route_download_content_to_documents", fake_route)

    docs = await loader.load()

    assert len(docs) == 3
    download_docs = [doc for doc in docs if doc.metadata.get("source_type") == "web_download"]
    assert len(download_docs) == 1
    assert download_docs[0].metadata["source"] == "https://example.com/report.pdf"
    assert "web_depth" in download_docs[0].metadata
    assert "start_url" in download_docs[0].metadata


@pytest.mark.asyncio
async def test_async_download_failures_are_reported_and_crawl_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        follow_download_links=True,
    )

    async def fake_fetch(url: str) -> AsyncFetchResult:
        if url == "https://example.com/root":
            return _async_html_result(
                url,
                (
                    '<html><body>'
                    '<a href="/ok">ok</a>'
                    '<a href="/file.bin">bin</a>'
                    "</body></html>"
                ),
            )
        if url == "https://example.com/ok":
            return _async_html_result(url, "<html><body><p>ok</p></body></html>")
        if url == "https://example.com/file.bin":
            return _async_download_result(
                url,
                "application/octet-stream",
                b"binary",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Disposition": 'attachment; filename="file.bin"',
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_fetch)

    docs = await loader.load()

    assert len(docs) == 2
    assert loader.last_stats["error_count"] >= 1
    assert any(err["stage"] == "download" for err in loader.last_errors)


@pytest.mark.asyncio
async def test_async_custom_link_extractor_accepts_awaitable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def extractor(_document, _url):
        return ["/custom"]

    loader = AsyncWebLoader(
        url="https://example.com/root",
        depth=1,
        fetch_mode="requests",
        custom_link_extractors=[extractor],
    )

    async def fake_fetch(url: str) -> AsyncFetchResult:
        if url == "https://example.com/root":
            return _async_html_result(url, "<html><body><p>root</p></body></html>")
        if url == "https://example.com/custom":
            return _async_html_result(url, "<html><body><p>custom</p></body></html>")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(loader, "_fetch_via_requests_async", fake_fetch)

    docs = await loader.load()

    sources = {doc.metadata["source"] for doc in docs}
    assert "https://example.com/root" in sources
    assert "https://example.com/custom" in sources
