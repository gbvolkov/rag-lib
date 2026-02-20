from __future__ import annotations

import asyncio

import pytest

from rag_lib.loaders.web_playwright_extractors import (
    PlaywrightExtractionConfig,
    PlaywrightNavigationConfig,
    PlaywrightProfileConfig,
    build_async_playwright_extractor,
    build_sync_playwright_extractor,
    get_playwright_profile_defaults,
    run_async_cleanup_navigation,
    run_async_playwright_extraction,
    run_sync_cleanup_navigation,
    run_sync_playwright_extraction,
)
from rag_lib.loaders.web_common import WebCleanupConfig, WebLink


class _SyncPage:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.wait_calls: list[int] = []

    def evaluate(self, _script, _arg=None):
        if not self.outputs:
            return []
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def wait_for_timeout(self, timeout_ms: int):
        self.wait_calls.append(timeout_ms)


class _AsyncPage:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.wait_calls: list[int] = []

    async def evaluate(self, _script, _arg=None):
        if not self.outputs:
            return []
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def wait_for_timeout(self, timeout_ms: int):
        self.wait_calls.append(timeout_ms)


def test_anchors_profile_extracts_links() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(PlaywrightProfileConfig(profile="anchors", selectors=("a[href]",)),)
    )
    page = _SyncPage(outputs=[["/a", "https://example.com/b"]])

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    assert {link.url for link in links} == {
        "https://example.com/a",
        "https://example.com/b",
    }


def test_attributes_profile_extracts_links() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="attributes",
                selectors=("[data-url]",),
                attributes=("data-url",),
            ),
        )
    )
    page = _SyncPage(outputs=[["/from-data", "https://example.com/x"]])

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    assert {link.url for link in links} == {
        "https://example.com/from-data",
        "https://example.com/x",
    }


def test_onclick_regex_profile_extracts_links() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="onclick_regex",
                selectors=("[onclick]",),
                attributes=("onclick",),
                regex_pattern=r"(?P<value>/[A-Za-z0-9_/-]+)",
            ),
        )
    )
    page = _SyncPage(outputs=[["go('/page-a')", "open('/page-b')", "noop"]])

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    assert {link.url for link in links} == {
        "https://example.com/page-a",
        "https://example.com/page-b",
    }


def test_eval_profile_accepts_dict_and_list_outputs() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="eval",
                script="() => []",
                is_navigation=True,
                source_tag="eval-profile",
                source_classes=("profile",),
            ),
        )
    )
    page = _SyncPage(
        outputs=[
            [
                "/one",
                {"url": "/two", "source_tag": "custom", "source_classes": ["x"]},
            ]
        ]
    )

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    url_map = {link.url: link for link in links}
    assert "https://example.com/one" in url_map
    assert "https://example.com/two" in url_map
    assert url_map["https://example.com/one"].is_navigation is True
    assert url_map["https://example.com/one"].source_tag == "eval-profile"
    assert url_map["https://example.com/one"].source_classes == ("profile",)
    assert url_map["https://example.com/two"].source_tag == "custom"
    assert url_map["https://example.com/two"].source_classes == ("x",)


def test_paginated_eval_profile_loops_until_no_new_links() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="paginated_eval",
                seed_script="() => true",
                extract_script="() => []",
                next_page_script="() => true",
                max_pages=5,
                wait_after_action_ms=100,
            ),
        )
    )
    page = _SyncPage(
        outputs=[
            True,  # seed
            ["/a"],  # extract page 1
            True,  # next page
            ["/a", "/b"],  # extract page 2
            True,  # next page
            ["/a", "/b"],  # extract page 3 -> no new links, stop
        ]
    )

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    assert {link.url for link in links} == {
        "https://example.com/a",
        "https://example.com/b",
    }
    assert page.wait_calls == [100, 100, 100]


def test_url_include_and_exclude_filters() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="eval",
                script="() => []",
                include_url_patterns=(r"/keep",),
                exclude_url_patterns=(r"skip",),
            ),
        )
    )
    page = _SyncPage(outputs=[["/keep-1", "/keep-skip", "/drop"]])

    links, errors = run_sync_playwright_extraction(
        config=config,
        page=page,
        base_url="https://example.com/root",
    )

    assert errors == []
    assert [link.url for link in links] == ["https://example.com/keep-1"]


@pytest.mark.asyncio
async def test_sync_and_async_runner_parity() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(PlaywrightProfileConfig(profile="eval", script="() => []"),)
    )
    sync_page = _SyncPage(outputs=[["/a", "/b"]])
    async_page = _AsyncPage(outputs=[["/a", "/b"]])

    sync_links, sync_errors = run_sync_playwright_extraction(
        config=config,
        page=sync_page,
        base_url="https://example.com/root",
    )
    async_links, async_errors = await run_async_playwright_extraction(
        config=config,
        page=async_page,
        base_url="https://example.com/root",
    )

    assert sync_errors == []
    assert async_errors == []
    assert {link.url for link in sync_links} == {link.url for link in async_links}


def test_sync_and_async_builders_return_extractors() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(PlaywrightProfileConfig(profile="eval", script="() => []"),)
    )
    sync_extractor = build_sync_playwright_extractor(config)
    async_extractor = build_async_playwright_extractor(config)

    sync_links = sync_extractor(_SyncPage(outputs=[["/a"]]), "https://example.com/root")
    assert {link.url for link in sync_links} == {"https://example.com/a"}

    async def _run_async():
        return await async_extractor(_AsyncPage(outputs=[["/a"]]), "https://example.com/root")

    async_links = asyncio.run(_run_async())
    assert {link.url for link in async_links} == {"https://example.com/a"}


def test_continue_on_error_false_raises() -> None:
    config = PlaywrightExtractionConfig(
        profiles=(PlaywrightProfileConfig(profile="eval", script="() => []"),),
        continue_on_error=False,
    )
    page = _SyncPage(outputs=[RuntimeError("boom")])

    with pytest.raises(RuntimeError):
        run_sync_playwright_extraction(
            config=config,
            page=page,
            base_url="https://example.com/root",
        )


def test_get_profile_defaults() -> None:
    anchors = get_playwright_profile_defaults("anchors")
    assert anchors.profile == "anchors"
    assert anchors.selectors == ("a[href]",)


class _NavigationSyncPage:
    def __init__(self, html_states: list[str], click_results: list[dict[str, object]]):
        self.html_states = html_states
        self.click_results = list(click_results)
        self.state_index = 0
        self.wait_calls: list[int] = []

    def evaluate(self, _script, arg=None):
        if isinstance(arg, dict) and "clickable_selectors" in arg:
            if not self.click_results:
                return {"clicked": False, "reason": "empty"}
            result = self.click_results.pop(0)
            if result.get("clicked") and self.state_index < len(self.html_states) - 1:
                self.state_index += 1
            return result
        if isinstance(arg, list):
            return True
        return []

    def content(self):
        return self.html_states[self.state_index]

    def wait_for_timeout(self, timeout_ms: int):
        self.wait_calls.append(timeout_ms)


class _NavigationAsyncPage:
    def __init__(self, html_states: list[str], click_results: list[dict[str, object]]):
        self.html_states = html_states
        self.click_results = list(click_results)
        self.state_index = 0
        self.wait_calls: list[int] = []

    async def evaluate(self, _script, arg=None):
        if isinstance(arg, dict) and "clickable_selectors" in arg:
            if not self.click_results:
                return {"clicked": False, "reason": "empty"}
            result = self.click_results.pop(0)
            if result.get("clicked") and self.state_index < len(self.html_states) - 1:
                self.state_index += 1
            return result
        if isinstance(arg, list):
            return True
        return []

    async def content(self):
        return self.html_states[self.state_index]

    async def wait_for_timeout(self, timeout_ms: int):
        self.wait_calls.append(timeout_ms)


def test_sync_cleanup_navigation_collects_states_and_callback_links() -> None:
    page = _NavigationSyncPage(
        html_states=[
            "<html><body>page-0</body></html>",
            "<html><body>page-1</body></html>",
            "<html><body>page-2</body></html>",
        ],
        click_results=[
            {"clicked": True, "reason": "ok"},
            {"clicked": True, "reason": "ok"},
            {"clicked": False, "reason": "no_candidate"},
        ],
    )

    def on_state(_page, base_url: str):
        return [WebLink(url=f"{base_url}/state")], []

    result = run_sync_cleanup_navigation(
        page=page,
        base_url="https://example.com/root",
        cleanup_config=WebCleanupConfig(navigation_texts=(">",)),
        config=PlaywrightNavigationConfig(max_clicks=5, max_states=5, wait_after_click_ms=0),
        on_state=on_state,
    )

    assert result.click_count == 2
    assert len(result.states) == 2
    assert all(state.extra_links for state in result.states)


@pytest.mark.asyncio
async def test_async_cleanup_navigation_collects_states() -> None:
    page = _NavigationAsyncPage(
        html_states=[
            "<html><body>a</body></html>",
            "<html><body>b</body></html>",
            "<html><body>c</body></html>",
        ],
        click_results=[
            {"clicked": True, "reason": "ok"},
            {"clicked": True, "reason": "ok"},
            {"clicked": False, "reason": "no_candidate"},
        ],
    )

    async def on_state(_page, base_url: str):
        return [WebLink(url=f"{base_url}/state")], []

    result = await run_async_cleanup_navigation(
        page=page,
        base_url="https://example.com/root",
        cleanup_config=WebCleanupConfig(navigation_texts=(">",)),
        config=PlaywrightNavigationConfig(max_clicks=5, max_states=5, wait_after_click_ms=0),
        on_state=on_state,
    )

    assert result.click_count == 2
    assert len(result.states) == 2


def test_sync_cleanup_navigation_stops_on_no_change_limit() -> None:
    page = _NavigationSyncPage(
        html_states=["<html><body>same</body></html>"],
        click_results=[
            {"clicked": True, "reason": "ok"},
            {"clicked": True, "reason": "ok"},
        ],
    )

    result = run_sync_cleanup_navigation(
        page=page,
        base_url="https://example.com/root",
        cleanup_config=WebCleanupConfig(navigation_texts=(">",)),
        config=PlaywrightNavigationConfig(
            max_clicks=5,
            max_states=5,
            wait_after_click_ms=0,
            state_change_timeout_ms=20,
            state_poll_interval_ms=5,
            max_no_change_clicks=0,
        ),
    )

    assert result.click_count == 1
    assert len(result.states) == 0


def test_cleanup_navigation_requires_markers() -> None:
    page = _NavigationSyncPage(
        html_states=["<html><body>a</body></html>", "<html><body>b</body></html>"],
        click_results=[{"clicked": True, "reason": "ok"}],
    )
    result = run_sync_cleanup_navigation(
        page=page,
        base_url="https://example.com/root",
        cleanup_config=WebCleanupConfig(),
        config=PlaywrightNavigationConfig(),
    )
    assert result.click_count == 0
    assert len(result.states) == 0
