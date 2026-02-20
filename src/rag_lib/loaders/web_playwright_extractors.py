from __future__ import annotations

import hashlib
import inspect
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Literal, Sequence

from rag_lib.loaders.web_common import (
    WebCleanupConfig,
    WebLink,
    WebLinkInput,
    merge_web_links,
    normalize_web_link_input,
)

PlaywrightProfileName = Literal["anchors", "attributes", "onclick_regex", "eval", "paginated_eval"]
NavigationStateDocumentMode = Literal["separate_documents", "single_document"]

SyncPlaywrightLinkExtractor = Callable[[Any, str], Sequence[WebLinkInput] | WebLinkInput | None]
AsyncPlaywrightLinkExtractor = Callable[
    [Any, str],
    Sequence[WebLinkInput] | Awaitable[Sequence[WebLinkInput]] | WebLinkInput | Awaitable[WebLinkInput] | None,
]
SyncPlaywrightStateCallback = Callable[[Any, str], tuple[list[WebLink], list[str]]]
AsyncPlaywrightStateCallback = Callable[
    [Any, str],
    tuple[list[WebLink], list[str]] | Awaitable[tuple[list[WebLink], list[str]]],
]


@dataclass(frozen=True)
class PlaywrightProfileConfig:
    profile: PlaywrightProfileName
    selectors: Sequence[str] = ()
    attributes: Sequence[str] = ()
    regex_pattern: str = ""
    url_template: str = "{value}"
    script: str = ""
    script_args: Any = None
    seed_script: str = ""
    next_page_script: str = ""
    extract_script: str = ""
    max_pages: int = 1
    wait_after_action_ms: int = 700
    include_url_patterns: Sequence[str] = ()
    exclude_url_patterns: Sequence[str] = ()
    is_navigation: bool = False
    source_tag: str = ""
    source_classes: Sequence[str] = ()


@dataclass(frozen=True)
class PlaywrightExtractionConfig:
    profiles: Sequence[PlaywrightProfileConfig] = ()
    continue_on_error: bool = True
    max_profile_runtime_ms: int | None = None


@dataclass(frozen=True)
class PlaywrightNavigationConfig:
    enabled: bool = True
    max_clicks: int = 20
    max_states: int = 25
    wait_after_click_ms: int = 800
    state_change_timeout_ms: int = 5000
    state_poll_interval_ms: int = 200
    max_no_change_clicks: int = 1
    clickable_selectors: Sequence[str] = ("a", "button", "[role='button']", "[onclick]")
    forward_text_markers: Sequence[str] = (">", "next", "›", "»", "→")
    backward_text_markers: Sequence[str] = ("<", "prev", "previous", "‹", "«", "←")
    content_ready_selectors: Sequence[str] = ()
    navigation_state_document_mode: NavigationStateDocumentMode = "separate_documents"


@dataclass(frozen=True)
class PlaywrightNavigationState:
    html: str
    content_hash: str
    click_count: int = 0
    extra_links: tuple[WebLink, ...] = ()
    extractor_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlaywrightNavigationRunResult:
    states: tuple[PlaywrightNavigationState, ...] = ()
    click_count: int = 0
    errors: tuple[str, ...] = ()


_JS_EXTRACT_VALUES_BY_SELECTORS = """
(args) => {
  const selectors = Array.isArray(args?.selectors) ? args.selectors : [];
  const attributes = Array.isArray(args?.attributes) ? args.attributes : [];
  const values = [];
  const seen = new Set();

  for (const selector of selectors) {
    let nodes = [];
    try {
      nodes = Array.from(document.querySelectorAll(selector));
    } catch (_error) {
      continue;
    }
    for (const node of nodes) {
      for (const attr of attributes) {
        const value = node.getAttribute?.(attr);
        if (typeof value !== "string") {
          continue;
        }
        const trimmed = value.trim();
        if (!trimmed) {
          continue;
        }
        const marker = `${attr}::${trimmed}`;
        if (seen.has(marker)) {
          continue;
        }
        seen.add(marker);
        values.push(trimmed);
      }
    }
  }

  return values;
}
"""


_JS_EXTRACT_TEXT_BY_SELECTORS = """
(args) => {
  const selectors = Array.isArray(args?.selectors) ? args.selectors : [];
  const values = [];
  const seen = new Set();

  for (const selector of selectors) {
    let nodes = [];
    try {
      nodes = Array.from(document.querySelectorAll(selector));
    } catch (_error) {
      continue;
    }
    for (const node of nodes) {
      const value = (node.textContent || "").trim();
      if (!value) {
        continue;
      }
      if (seen.has(value)) {
        continue;
      }
      seen.add(value);
      values.push(value);
    }
  }

  return values;
}
"""

_JS_HAS_READY_SELECTORS = """
(selectors) => {
  if (!Array.isArray(selectors) || selectors.length === 0) {
    return true;
  }
  for (const selector of selectors) {
    try {
      if (document.querySelector(selector)) {
        return true;
      }
    } catch (_error) {
      continue;
    }
  }
  return false;
}
"""

_JS_CLICK_BEST_NAV_CANDIDATE = """
(args) => {
  const clickableSelectors = Array.isArray(args?.clickable_selectors) ? args.clickable_selectors : [];
  const classRules = Array.isArray(args?.navigation_classes) ? args.navigation_classes : [];
  const styleRules = Array.isArray(args?.navigation_styles) ? args.navigation_styles : [];
  const textRules = Array.isArray(args?.navigation_texts) ? args.navigation_texts : [];
  const forwardMarkers = Array.isArray(args?.forward_text_markers) ? args.forward_text_markers : [];
  const backwardMarkers = Array.isArray(args?.backward_text_markers) ? args.backward_text_markers : [];

  const normalizeText = (value) => (value || "").replace(/\\s+/g, " ").trim();
  const normalizeStyle = (value) => (value || "").replace(/\\s+/g, "").toLowerCase();

  const ruleTokens = classRules.map((rule) => String(rule || "").trim().split(/\\s+/).filter(Boolean)).filter((parts) => parts.length > 0);
  const normalizedStyles = styleRules.map((rule) => normalizeStyle(rule)).filter(Boolean);
  const markerSet = new Set(textRules.map((value) => normalizeText(String(value || ""))).filter(Boolean));
  const forwardSet = new Set(forwardMarkers.map((value) => normalizeText(String(value || "")).toLowerCase()).filter(Boolean));
  const backwardSet = new Set(backwardMarkers.map((value) => normalizeText(String(value || "")).toLowerCase()).filter(Boolean));

  if (clickableSelectors.length === 0) {
    return { clicked: false, reason: "no_clickable_selectors" };
  }

  const clickable = [];
  const seen = new Set();
  for (const selector of clickableSelectors) {
    let nodes = [];
    try {
      nodes = Array.from(document.querySelectorAll(selector));
    } catch (_error) {
      continue;
    }
    for (const node of nodes) {
      if (!node || seen.has(node)) {
        continue;
      }
      seen.add(node);
      clickable.push(node);
    }
  }

  const classRuleMatches = (el) => {
    if (ruleTokens.length === 0) {
      return false;
    }
    let current = el;
    while (current) {
      const classes = new Set(String(current.className || "").split(/\\s+/).filter(Boolean));
      if (classes.size > 0) {
        for (const parts of ruleTokens) {
          let ok = true;
          for (const token of parts) {
            if (!classes.has(token)) {
              ok = false;
              break;
            }
          }
          if (ok) {
            return true;
          }
        }
      }
      current = current.parentElement;
    }
    return false;
  };

  const styleRuleMatches = (el) => {
    if (normalizedStyles.length === 0) {
      return false;
    }
    let current = el;
    while (current) {
      const styleAttr = normalizeStyle(current.getAttribute?.("style") || "");
      if (styleAttr) {
        for (const rule of normalizedStyles) {
          if (rule && styleAttr.includes(rule)) {
            return true;
          }
        }
      }
      current = current.parentElement;
    }
    return false;
  };

  const isVisible = (el) => {
    if (!el || !el.isConnected) {
      return false;
    }
    const style = window.getComputedStyle(el);
    if (!style) {
      return false;
    }
    if (style.display === "none" || style.visibility === "hidden" || style.pointerEvents === "none" || Number(style.opacity || "1") <= 0) {
      return false;
    }
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return false;
    }
    return true;
  };

  const candidates = [];
  for (const el of clickable) {
    if (!isVisible(el)) {
      continue;
    }
    if (el.disabled || el.getAttribute?.("aria-disabled") === "true") {
      continue;
    }
    const text = normalizeText(el.textContent || el.innerText || el.getAttribute?.("aria-label") || "");
    const lower = text.toLowerCase();
    const textMatch = markerSet.has(text);
    const classMatch = classRuleMatches(el);
    const styleMatch = styleRuleMatches(el);
    if (!textMatch && !classMatch && !styleMatch) {
      continue;
    }

    let score = 0;
    if (classMatch) {
      score += 10;
    }
    if (styleMatch) {
      score += 10;
    }
    if (textMatch) {
      score += 20;
    }
    if (forwardSet.has(lower)) {
      score += 100;
    }
    if (backwardSet.has(lower)) {
      score -= 100;
    }

    const rel = String(el.getAttribute?.("rel") || "").toLowerCase();
    if (rel.includes("next")) {
      score += 80;
    }
    const aria = String(el.getAttribute?.("aria-label") || "").toLowerCase();
    if (aria.includes("next")) {
      score += 60;
    }
    if (aria.includes("prev")) {
      score -= 60;
    }

    const href = String(el.getAttribute?.("href") || "");
    if (href.trim()) {
      score += 5;
    }

    const signature = `${String(el.tagName || "").toLowerCase()}|${String(el.className || "")}|${text}|${href}|${String(el.getAttribute?.("style") || "")}`;
    candidates.push({ el, score, text, signature });
  }

  if (candidates.length === 0) {
    return { clicked: false, reason: "no_candidate" };
  }

  candidates.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    return left.text.length - right.text.length;
  });

  const target = candidates[0];
  try {
    target.el.click();
  } catch (_error) {
    try {
      target.el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
    } catch (error) {
      return { clicked: false, reason: `click_failed:${String(error)}` };
    }
  }

  return {
    clicked: true,
    reason: "ok",
    signature: target.signature,
    text: target.text,
    score: target.score,
  };
}
"""


def get_playwright_profile_defaults(name: str) -> PlaywrightProfileConfig:
    normalized = str(name).strip()
    if normalized == "anchors":
        return PlaywrightProfileConfig(profile="anchors", selectors=("a[href]",), source_tag="anchors")
    if normalized == "attributes":
        return PlaywrightProfileConfig(
            profile="attributes",
            selectors=("[data-url], [data-href], [href]",),
            attributes=("data-url", "data-href", "href"),
            source_tag="attributes",
        )
    if normalized == "onclick_regex":
        return PlaywrightProfileConfig(
            profile="onclick_regex",
            selectors=("[onclick]",),
            attributes=("onclick",),
            regex_pattern=r"(?P<value>https?://[^'\"\\s)]+|/[^'\"\\s)]+)",
            source_tag="onclick_regex",
        )
    if normalized == "eval":
        return PlaywrightProfileConfig(profile="eval", source_tag="eval")
    if normalized == "paginated_eval":
        return PlaywrightProfileConfig(profile="paginated_eval", max_pages=3, source_tag="paginated_eval")
    raise ValueError(f"Unknown playwright profile default: {name}")


def build_sync_playwright_extractor(config: PlaywrightExtractionConfig) -> SyncPlaywrightLinkExtractor:
    prepared = _prepare_extraction_config(config)

    def extractor(page: Any, base_url: str) -> list[WebLink]:
        links, _errors = run_sync_playwright_extraction(config=prepared, page=page, base_url=base_url)
        return links

    return extractor


def build_async_playwright_extractor(config: PlaywrightExtractionConfig) -> AsyncPlaywrightLinkExtractor:
    prepared = _prepare_extraction_config(config)

    async def extractor(page: Any, base_url: str) -> list[WebLink]:
        links, _errors = await run_async_playwright_extraction(config=prepared, page=page, base_url=base_url)
        return links

    return extractor


def compose_sync_playwright_link_extractors(
    extractors: Sequence[SyncPlaywrightLinkExtractor],
) -> SyncPlaywrightLinkExtractor:
    extractor_list = list(extractors)

    def composed(page: Any, base_url: str) -> list[WebLinkInput]:
        merged: list[WebLinkInput] = []
        for extractor in extractor_list:
            current = extractor(page, base_url)
            if current is None:
                continue
            merged.extend(_coerce_output_items(current))
        return merged

    return composed


def compose_async_playwright_link_extractors(
    extractors: Sequence[AsyncPlaywrightLinkExtractor],
) -> AsyncPlaywrightLinkExtractor:
    extractor_list = list(extractors)

    async def composed(page: Any, base_url: str) -> list[WebLinkInput]:
        merged: list[WebLinkInput] = []
        for extractor in extractor_list:
            current = extractor(page, base_url)
            if inspect.isawaitable(current):
                current = await current
            if current is None:
                continue
            merged.extend(_coerce_output_items(current))
        return merged

    return composed


def run_sync_playwright_extraction(
    *,
    config: PlaywrightExtractionConfig,
    page: Any,
    base_url: str,
) -> tuple[list[WebLink], list[str]]:
    prepared = _prepare_extraction_config(config)
    start = time.monotonic()

    errors: list[str] = []
    links: list[WebLink] = []
    for profile in prepared.profiles:
        timeout_error = _check_profile_timeout(prepared, start)
        if timeout_error is not None:
            errors.append(timeout_error)
            break

        try:
            raw = _execute_profile_sync(page=page, profile=profile)
            links.extend(_normalize_profile_output(raw, profile=profile, base_url=base_url))
        except Exception as exc:
            message = f"{profile.profile}: {exc}"
            if prepared.continue_on_error:
                errors.append(message)
                continue
            raise RuntimeError(message) from exc

    return merge_web_links(links), errors


async def run_async_playwright_extraction(
    *,
    config: PlaywrightExtractionConfig,
    page: Any,
    base_url: str,
) -> tuple[list[WebLink], list[str]]:
    prepared = _prepare_extraction_config(config)
    start = time.monotonic()

    errors: list[str] = []
    links: list[WebLink] = []
    for profile in prepared.profiles:
        timeout_error = _check_profile_timeout(prepared, start)
        if timeout_error is not None:
            errors.append(timeout_error)
            break

        try:
            raw = await _execute_profile_async(page=page, profile=profile)
            links.extend(_normalize_profile_output(raw, profile=profile, base_url=base_url))
        except Exception as exc:
            message = f"{profile.profile}: {exc}"
            if prepared.continue_on_error:
                errors.append(message)
                continue
            raise RuntimeError(message) from exc

    return merge_web_links(links), errors


def run_sync_cleanup_navigation(
    *,
    page: Any,
    base_url: str,
    cleanup_config: WebCleanupConfig | None,
    config: PlaywrightNavigationConfig | None = None,
    on_state: SyncPlaywrightStateCallback | None = None,
) -> PlaywrightNavigationRunResult:
    prepared = _prepare_navigation_config(config)
    if not _should_run_navigation(cleanup_config=cleanup_config, config=prepared):
        return PlaywrightNavigationRunResult()

    marker_payload = _navigation_marker_payload(cleanup_config=cleanup_config, config=prepared)
    errors: list[str] = []
    states: list[PlaywrightNavigationState] = []
    seen_hashes: set[str] = set()
    click_count = 0
    no_change_clicks = 0

    previous_html = page.content() or ""
    previous_hash = _content_hash(previous_html)
    seen_hashes.add(previous_hash)
    total_states = 1

    for _ in range(prepared.max_clicks):
        if total_states >= prepared.max_states:
            break

        try:
            click_result = _evaluate_sync(page, _JS_CLICK_BEST_NAV_CANDIDATE, marker_payload)
        except Exception as exc:
            errors.append(f"click_candidate: {exc}")
            break

        if not isinstance(click_result, dict) or not click_result.get("clicked"):
            break
        click_count += 1

        changed, current_html, current_hash = _wait_for_changed_content_sync(
            page=page,
            previous_hash=previous_hash,
            config=prepared,
        )
        if not changed:
            no_change_clicks += 1
            if no_change_clicks > prepared.max_no_change_clicks:
                break
            continue

        previous_hash = current_hash
        if current_hash in seen_hashes:
            no_change_clicks += 1
            if no_change_clicks > prepared.max_no_change_clicks:
                break
            continue

        no_change_clicks = 0
        seen_hashes.add(current_hash)
        total_states += 1

        if prepared.wait_after_click_ms > 0:
            page.wait_for_timeout(prepared.wait_after_click_ms)
            current_html = page.content() or current_html
            current_hash = _content_hash(current_html)
            previous_hash = current_hash

        extra_links: list[WebLink] = []
        extractor_errors: list[str] = []
        if on_state is not None:
            try:
                extra_links, extractor_errors = on_state(page, base_url)
            except Exception as exc:
                extractor_errors = [f"state_callback: {exc}"]

        states.append(
            PlaywrightNavigationState(
                html=current_html,
                content_hash=current_hash,
                click_count=click_count,
                extra_links=tuple(extra_links),
                extractor_errors=tuple(extractor_errors),
            )
        )

    return PlaywrightNavigationRunResult(
        states=tuple(states),
        click_count=click_count,
        errors=tuple(errors),
    )


async def run_async_cleanup_navigation(
    *,
    page: Any,
    base_url: str,
    cleanup_config: WebCleanupConfig | None,
    config: PlaywrightNavigationConfig | None = None,
    on_state: AsyncPlaywrightStateCallback | None = None,
) -> PlaywrightNavigationRunResult:
    prepared = _prepare_navigation_config(config)
    if not _should_run_navigation(cleanup_config=cleanup_config, config=prepared):
        return PlaywrightNavigationRunResult()

    marker_payload = _navigation_marker_payload(cleanup_config=cleanup_config, config=prepared)
    errors: list[str] = []
    states: list[PlaywrightNavigationState] = []
    seen_hashes: set[str] = set()
    click_count = 0
    no_change_clicks = 0

    previous_html = await page.content() or ""
    previous_hash = _content_hash(previous_html)
    seen_hashes.add(previous_hash)
    total_states = 1

    for _ in range(prepared.max_clicks):
        if total_states >= prepared.max_states:
            break

        try:
            click_result = await _evaluate_async(page, _JS_CLICK_BEST_NAV_CANDIDATE, marker_payload)
        except Exception as exc:
            errors.append(f"click_candidate: {exc}")
            break

        if not isinstance(click_result, dict) or not click_result.get("clicked"):
            break
        click_count += 1

        changed, current_html, current_hash = await _wait_for_changed_content_async(
            page=page,
            previous_hash=previous_hash,
            config=prepared,
        )
        if not changed:
            no_change_clicks += 1
            if no_change_clicks > prepared.max_no_change_clicks:
                break
            continue

        previous_hash = current_hash
        if current_hash in seen_hashes:
            no_change_clicks += 1
            if no_change_clicks > prepared.max_no_change_clicks:
                break
            continue

        no_change_clicks = 0
        seen_hashes.add(current_hash)
        total_states += 1

        if prepared.wait_after_click_ms > 0:
            await page.wait_for_timeout(prepared.wait_after_click_ms)
            current_html = await page.content() or current_html
            current_hash = _content_hash(current_html)
            previous_hash = current_hash

        extra_links: list[WebLink] = []
        extractor_errors: list[str] = []
        if on_state is not None:
            try:
                state_result = on_state(page, base_url)
                if inspect.isawaitable(state_result):
                    state_result = await state_result
                extra_links, extractor_errors = state_result
            except Exception as exc:
                extractor_errors = [f"state_callback: {exc}"]

        states.append(
            PlaywrightNavigationState(
                html=current_html,
                content_hash=current_hash,
                click_count=click_count,
                extra_links=tuple(extra_links),
                extractor_errors=tuple(extractor_errors),
            )
        )

    return PlaywrightNavigationRunResult(
        states=tuple(states),
        click_count=click_count,
        errors=tuple(errors),
    )


def _normalize_navigation_marker(value: str) -> str:
    return " ".join(str(value or "").split())


def _content_hash(html: str) -> str:
    return hashlib.md5((html or "").encode("utf-8")).hexdigest()


def _prepare_navigation_config(config: PlaywrightNavigationConfig | None) -> PlaywrightNavigationConfig:
    value = config or PlaywrightNavigationConfig()
    if value.max_clicks <= 0:
        raise ValueError("playwright navigation max_clicks must be > 0.")
    if value.max_states <= 0:
        raise ValueError("playwright navigation max_states must be > 0.")
    if value.wait_after_click_ms < 0:
        raise ValueError("playwright navigation wait_after_click_ms must be >= 0.")
    if value.state_change_timeout_ms <= 0:
        raise ValueError("playwright navigation state_change_timeout_ms must be > 0.")
    if value.state_poll_interval_ms <= 0:
        raise ValueError("playwright navigation state_poll_interval_ms must be > 0.")
    if value.max_no_change_clicks < 0:
        raise ValueError("playwright navigation max_no_change_clicks must be >= 0.")
    if value.navigation_state_document_mode not in {"separate_documents", "single_document"}:
        raise ValueError("navigation_state_document_mode must be separate_documents or single_document.")

    clickable_selectors = tuple(item.strip() for item in value.clickable_selectors if str(item).strip())
    if not clickable_selectors:
        raise ValueError("playwright navigation clickable_selectors must be non-empty.")

    forward_text_markers = tuple(
        _normalize_navigation_marker(str(item)).lower()
        for item in value.forward_text_markers
        if _normalize_navigation_marker(str(item))
    )
    backward_text_markers = tuple(
        _normalize_navigation_marker(str(item)).lower()
        for item in value.backward_text_markers
        if _normalize_navigation_marker(str(item))
    )
    content_ready_selectors = tuple(item.strip() for item in value.content_ready_selectors if str(item).strip())

    return replace(
        value,
        clickable_selectors=clickable_selectors,
        forward_text_markers=forward_text_markers,
        backward_text_markers=backward_text_markers,
        content_ready_selectors=content_ready_selectors,
    )


def _should_run_navigation(
    *,
    cleanup_config: WebCleanupConfig | None,
    config: PlaywrightNavigationConfig,
) -> bool:
    if not config.enabled:
        return False
    if cleanup_config is None:
        return False
    return any(
        (
            bool(cleanup_config.navigation_classes),
            bool(cleanup_config.navigation_styles),
            bool(cleanup_config.navigation_texts),
        )
    )


def _navigation_marker_payload(
    *,
    cleanup_config: WebCleanupConfig | None,
    config: PlaywrightNavigationConfig,
) -> dict[str, Any]:
    cleanup = cleanup_config or WebCleanupConfig()
    navigation_classes = tuple(str(item).strip() for item in cleanup.navigation_classes if str(item).strip())
    navigation_styles = tuple(
        re.sub(r"\s+", "", str(item).lower())
        for item in cleanup.navigation_styles
        if str(item).strip()
    )
    navigation_texts = tuple(
        _normalize_navigation_marker(str(item))
        for item in cleanup.navigation_texts
        if _normalize_navigation_marker(str(item))
    )
    return {
        "clickable_selectors": list(config.clickable_selectors),
        "navigation_classes": list(navigation_classes),
        "navigation_styles": list(navigation_styles),
        "navigation_texts": list(navigation_texts),
        "forward_text_markers": list(config.forward_text_markers),
        "backward_text_markers": list(config.backward_text_markers),
    }


def _is_ready_sync(page: Any, selectors: Sequence[str]) -> bool:
    if not selectors:
        return True
    try:
        return bool(_evaluate_sync(page, _JS_HAS_READY_SELECTORS, list(selectors)))
    except Exception:
        return False


async def _is_ready_async(page: Any, selectors: Sequence[str]) -> bool:
    if not selectors:
        return True
    try:
        return bool(await _evaluate_async(page, _JS_HAS_READY_SELECTORS, list(selectors)))
    except Exception:
        return False


def _wait_for_changed_content_sync(
    *,
    page: Any,
    previous_hash: str,
    config: PlaywrightNavigationConfig,
) -> tuple[bool, str, str]:
    deadline = time.monotonic() + (config.state_change_timeout_ms / 1000.0)
    last_html = ""
    last_hash = previous_hash
    while time.monotonic() <= deadline:
        current_html = page.content() or ""
        current_hash = _content_hash(current_html)
        ready = _is_ready_sync(page, config.content_ready_selectors)
        last_html = current_html
        last_hash = current_hash
        if current_hash != previous_hash and ready:
            return True, current_html, current_hash
        page.wait_for_timeout(config.state_poll_interval_ms)
    return False, last_html, last_hash


async def _wait_for_changed_content_async(
    *,
    page: Any,
    previous_hash: str,
    config: PlaywrightNavigationConfig,
) -> tuple[bool, str, str]:
    deadline = time.monotonic() + (config.state_change_timeout_ms / 1000.0)
    last_html = ""
    last_hash = previous_hash
    while time.monotonic() <= deadline:
        current_html = await page.content() or ""
        current_hash = _content_hash(current_html)
        ready = await _is_ready_async(page, config.content_ready_selectors)
        last_html = current_html
        last_hash = current_hash
        if current_hash != previous_hash and ready:
            return True, current_html, current_hash
        await page.wait_for_timeout(config.state_poll_interval_ms)
    return False, last_html, last_hash


def _prepare_extraction_config(config: PlaywrightExtractionConfig) -> PlaywrightExtractionConfig:
    if config.max_profile_runtime_ms is not None and config.max_profile_runtime_ms <= 0:
        raise ValueError("max_profile_runtime_ms must be > 0 when provided.")

    profiles: list[PlaywrightProfileConfig] = []
    for profile in config.profiles:
        profiles.append(_prepare_profile(profile))
    return PlaywrightExtractionConfig(
        profiles=tuple(profiles),
        continue_on_error=config.continue_on_error,
        max_profile_runtime_ms=config.max_profile_runtime_ms,
    )


def _prepare_profile(profile: PlaywrightProfileConfig) -> PlaywrightProfileConfig:
    if profile.profile not in {"anchors", "attributes", "onclick_regex", "eval", "paginated_eval"}:
        raise ValueError(f"Unknown playwright profile: {profile.profile}")

    if profile.max_pages <= 0:
        raise ValueError("max_pages must be > 0.")
    if profile.wait_after_action_ms < 0:
        raise ValueError("wait_after_action_ms must be >= 0.")

    selectors = tuple(str(item).strip() for item in profile.selectors if str(item).strip())
    attributes = tuple(str(item).strip() for item in profile.attributes if str(item).strip())
    include_patterns = tuple(str(item).strip() for item in profile.include_url_patterns if str(item).strip())
    exclude_patterns = tuple(str(item).strip() for item in profile.exclude_url_patterns if str(item).strip())
    source_classes = tuple(str(item).strip() for item in profile.source_classes if str(item).strip())

    prepared = replace(
        profile,
        selectors=selectors,
        attributes=attributes,
        include_url_patterns=include_patterns,
        exclude_url_patterns=exclude_patterns,
        source_classes=source_classes,
    )

    if prepared.profile == "anchors":
        if not prepared.selectors:
            prepared = replace(prepared, selectors=("a[href]",))
        return prepared

    if prepared.profile == "attributes":
        if not prepared.selectors:
            raise ValueError("attributes profile requires selectors.")
        if not prepared.attributes:
            raise ValueError("attributes profile requires attributes.")
        return prepared

    if prepared.profile == "onclick_regex":
        if not prepared.selectors:
            raise ValueError("onclick_regex profile requires selectors.")
        if not prepared.regex_pattern.strip():
            raise ValueError("onclick_regex profile requires regex_pattern.")
        _ = re.compile(prepared.regex_pattern)
        return prepared

    if prepared.profile == "eval":
        if not prepared.script.strip():
            raise ValueError("eval profile requires script.")
        return prepared

    if prepared.profile == "paginated_eval":
        if not prepared.extract_script.strip():
            raise ValueError("paginated_eval profile requires extract_script.")
        return prepared

    raise ValueError(f"Unknown playwright profile: {prepared.profile}")


def _execute_profile_sync(*, page: Any, profile: PlaywrightProfileConfig) -> Any:
    if profile.profile == "anchors":
        return _evaluate_sync(
            page,
            _JS_EXTRACT_VALUES_BY_SELECTORS,
            {"selectors": list(profile.selectors), "attributes": ["href"]},
        )

    if profile.profile == "attributes":
        return _evaluate_sync(
            page,
            _JS_EXTRACT_VALUES_BY_SELECTORS,
            {"selectors": list(profile.selectors), "attributes": list(profile.attributes)},
        )

    if profile.profile == "onclick_regex":
        values: list[str]
        if profile.attributes:
            values = _evaluate_sync(
                page,
                _JS_EXTRACT_VALUES_BY_SELECTORS,
                {"selectors": list(profile.selectors), "attributes": list(profile.attributes)},
            )
        else:
            values = _evaluate_sync(
                page,
                _JS_EXTRACT_TEXT_BY_SELECTORS,
                {"selectors": list(profile.selectors)},
            )
        return _apply_regex_mapping(values=values, profile=profile)

    if profile.profile == "eval":
        return _evaluate_sync(page, profile.script, profile.script_args)

    if profile.profile == "paginated_eval":
        return _run_paginated_eval_sync(page=page, profile=profile)

    raise ValueError(f"Unknown profile: {profile.profile}")


async def _execute_profile_async(*, page: Any, profile: PlaywrightProfileConfig) -> Any:
    if profile.profile == "anchors":
        return await _evaluate_async(
            page,
            _JS_EXTRACT_VALUES_BY_SELECTORS,
            {"selectors": list(profile.selectors), "attributes": ["href"]},
        )

    if profile.profile == "attributes":
        return await _evaluate_async(
            page,
            _JS_EXTRACT_VALUES_BY_SELECTORS,
            {"selectors": list(profile.selectors), "attributes": list(profile.attributes)},
        )

    if profile.profile == "onclick_regex":
        values: list[str]
        if profile.attributes:
            values = await _evaluate_async(
                page,
                _JS_EXTRACT_VALUES_BY_SELECTORS,
                {"selectors": list(profile.selectors), "attributes": list(profile.attributes)},
            )
        else:
            values = await _evaluate_async(
                page,
                _JS_EXTRACT_TEXT_BY_SELECTORS,
                {"selectors": list(profile.selectors)},
            )
        return _apply_regex_mapping(values=values, profile=profile)

    if profile.profile == "eval":
        return await _evaluate_async(page, profile.script, profile.script_args)

    if profile.profile == "paginated_eval":
        return await _run_paginated_eval_async(page=page, profile=profile)

    raise ValueError(f"Unknown profile: {profile.profile}")


def _evaluate_sync(page: Any, script: str, arg: Any) -> Any:
    if arg is None:
        return page.evaluate(script)
    return page.evaluate(script, arg)


async def _evaluate_async(page: Any, script: str, arg: Any) -> Any:
    if arg is None:
        return await page.evaluate(script)
    return await page.evaluate(script, arg)


def _run_paginated_eval_sync(*, page: Any, profile: PlaywrightProfileConfig) -> list[Any]:
    if profile.seed_script:
        _evaluate_sync(page, profile.seed_script, profile.script_args)
        if profile.wait_after_action_ms > 0:
            page.wait_for_timeout(profile.wait_after_action_ms)

    seen: set[str] = set()
    collected: list[Any] = []
    for index in range(max(1, profile.max_pages)):
        raw = _evaluate_sync(page, profile.extract_script, profile.script_args)
        items = _coerce_output_items(raw)
        collected.extend(items)

        new_count = 0
        for token in _coerce_items_to_raw_url_tokens(items):
            if token in seen:
                continue
            seen.add(token)
            new_count += 1

        if index >= max(1, profile.max_pages) - 1:
            break
        if not profile.next_page_script:
            break
        if index > 0 and new_count == 0:
            break

        _evaluate_sync(page, profile.next_page_script, profile.script_args)
        if profile.wait_after_action_ms > 0:
            page.wait_for_timeout(profile.wait_after_action_ms)

    return collected


async def _run_paginated_eval_async(*, page: Any, profile: PlaywrightProfileConfig) -> list[Any]:
    if profile.seed_script:
        await _evaluate_async(page, profile.seed_script, profile.script_args)
        if profile.wait_after_action_ms > 0:
            await page.wait_for_timeout(profile.wait_after_action_ms)

    seen: set[str] = set()
    collected: list[Any] = []
    for index in range(max(1, profile.max_pages)):
        raw = await _evaluate_async(page, profile.extract_script, profile.script_args)
        items = _coerce_output_items(raw)
        collected.extend(items)

        new_count = 0
        for token in _coerce_items_to_raw_url_tokens(items):
            if token in seen:
                continue
            seen.add(token)
            new_count += 1

        if index >= max(1, profile.max_pages) - 1:
            break
        if not profile.next_page_script:
            break
        if index > 0 and new_count == 0:
            break

        await _evaluate_async(page, profile.next_page_script, profile.script_args)
        if profile.wait_after_action_ms > 0:
            await page.wait_for_timeout(profile.wait_after_action_ms)

    return collected


def _apply_regex_mapping(*, values: Sequence[str], profile: PlaywrightProfileConfig) -> list[str]:
    regex = re.compile(profile.regex_pattern)
    mapped: list[str] = []
    for value in values:
        for match in regex.finditer(str(value)):
            match_value: str
            if "value" in match.groupdict():
                match_value = match.group("value")
            elif match.groups():
                match_value = match.group(1)
            else:
                match_value = match.group(0)
            mapped.append(profile.url_template.format(value=match_value))
    return mapped


def _normalize_profile_output(
    output: Any,
    *,
    profile: PlaywrightProfileConfig,
    base_url: str,
) -> list[WebLink]:
    normalized_links: list[WebLink] = []
    for item in _coerce_output_items(output):
        explicit_nav: bool | None = None
        explicit_tag: str | None = None
        explicit_classes: tuple[str, ...] | None = None
        candidate: WebLinkInput

        if isinstance(item, dict):
            if "url" not in item:
                raise ValueError("Dictionary output must contain 'url'.")
            explicit_nav = bool(item["is_navigation"]) if "is_navigation" in item else None
            explicit_tag = str(item["source_tag"]).strip() if "source_tag" in item else None
            if "source_classes" in item and item["source_classes"] is not None:
                explicit_classes = tuple(str(v).strip() for v in item["source_classes"] if str(v).strip())
            candidate = str(item["url"])
        else:
            candidate = item
            if isinstance(item, WebLink):
                explicit_nav = item.is_navigation
                explicit_tag = item.source_tag
                explicit_classes = tuple(item.source_classes)

        resolved = normalize_web_link_input(candidate, base_url=base_url)
        if resolved is None:
            continue

        if not _matches_url_filters(
            resolved.url,
            include_patterns=profile.include_url_patterns,
            exclude_patterns=profile.exclude_url_patterns,
        ):
            continue

        source_classes = explicit_classes if explicit_classes else resolved.source_classes
        if not source_classes:
            source_classes = tuple(profile.source_classes)

        source_tag = explicit_tag if explicit_tag else resolved.source_tag
        if not source_tag:
            source_tag = profile.source_tag or profile.profile

        is_navigation = explicit_nav if explicit_nav is not None else resolved.is_navigation
        if not is_navigation and profile.is_navigation:
            is_navigation = True

        normalized_links.append(
            WebLink(
                url=resolved.url,
                source_classes=source_classes,
                is_navigation=is_navigation,
                source_tag=source_tag,
            )
        )

    return normalized_links


def _coerce_items_to_raw_url_tokens(items: Sequence[Any]) -> list[str]:
    tokens: list[str] = []
    for item in items:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                tokens.append(cleaned)
            continue
        if isinstance(item, WebLink):
            cleaned = item.url.strip()
            if cleaned:
                tokens.append(cleaned)
            continue
        if isinstance(item, tuple) and len(item) == 2:
            cleaned = str(item[0]).strip()
            if cleaned:
                tokens.append(cleaned)
            continue
        if isinstance(item, dict) and "url" in item:
            cleaned = str(item["url"]).strip()
            if cleaned:
                tokens.append(cleaned)
    return tokens


def _coerce_output_items(output: Any) -> list[Any]:
    if output is None:
        return []
    if isinstance(output, (str, WebLink, tuple, dict)):
        return [output]
    if isinstance(output, Sequence):
        return list(output)
    raise TypeError(f"Unsupported extractor output type: {type(output)}")


def _matches_url_filters(url: str, *, include_patterns: Sequence[str], exclude_patterns: Sequence[str]) -> bool:
    if include_patterns:
        if not any(re.search(pattern, url) for pattern in include_patterns):
            return False
    if exclude_patterns:
        if any(re.search(pattern, url) for pattern in exclude_patterns):
            return False
    return True


def _check_profile_timeout(
    config: PlaywrightExtractionConfig,
    start_monotonic: float,
) -> str | None:
    if config.max_profile_runtime_ms is None:
        return None
    elapsed_ms = int((time.monotonic() - start_monotonic) * 1000)
    if elapsed_ms <= config.max_profile_runtime_ms:
        return None
    return f"profile_chain_timeout: exceeded {config.max_profile_runtime_ms}ms"
