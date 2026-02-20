import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.loaders import (
    AsyncWebLoader,
    PlaywrightExtractionConfig,
    PlaywrightNavigationConfig,
    PlaywrightProfileConfig,
    WebCleanupConfig,
    WebLoader,
)


TARGET_URL = "https://plantpad.samlab.cn/search.html"
PLANTPAD_SEARCH_KEYWORD = ""# black spot"
PLANTPAD_MAX_RESULT_PAGES = 512

CLEANUP_CONFIG = WebCleanupConfig(
    duplicate_tags=("div", "p", "table"),
    non_recursive_classes=("tag",),
    navigation_classes=("menus",),
    navigation_styles=(),
    navigation_texts=("<", ">"),
    ignored_classes=("header",),
)

PLANTPAD_SEED_SCRIPT = """
async ({ keyword }) => {
  const current = new URL(window.location.href);
  if (!current.pathname.endsWith("/search.html")) {
    return false;
  }

  const app = document.querySelector("#app");
  const vm = app && app.__vue__;
  if (!vm) {
    return false;
  }

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  if ((!Array.isArray(vm.search_result) || vm.search_result.length === 0) && typeof vm.doSearch === "function") {
    if (keyword && typeof vm.search_context === "string" && !vm.search_context.trim()) {
      vm.search_context = keyword;
      const searchInput = document.querySelector(".search_input");
      if (searchInput) {
        searchInput.value = keyword;
      }
    }
    vm.page = 0;
    vm.doSearch();
    await sleep(900);
  }

  if ((!Array.isArray(vm.search_result) || vm.search_result.length === 0) && typeof vm.getSearchData === "function") {
    vm.page = 0;
    vm.getSearchData();
    await sleep(900);
  }

  return true;
}
"""


PLANTPAD_EXTRACT_SCRIPT = """
() => {
  const current = new URL(window.location.href);
  if (!current.pathname.endsWith("/search.html")) {
    return [];
  }
  const app = document.querySelector("#app");
  const vm = app && app.__vue__;
  if (!vm || !Array.isArray(vm.search_result)) {
    return [];
  }

  const urls = [];
  for (const item of vm.search_result) {
    const rawId = item && (item.img_id ?? item.imgId ?? item.id);
    if (rawId === undefined || rawId === null || rawId === "") {
      continue;
    }
    urls.push(`disease.html?img_id=${encodeURIComponent(String(rawId))}`);
  }
  return urls;
}
"""


PLANTPAD_NEXT_PAGE_SCRIPT = """
() => {
  const current = new URL(window.location.href);
  if (!current.pathname.endsWith("/search.html")) {
    return false;
  }
  const app = document.querySelector("#app");
  const vm = app && app.__vue__;
  if (!vm || typeof vm.nextPage !== "function") {
    return false;
  }
  if (vm.top) {
    return false;
  }
  const before = Number(vm.page || 0);
  vm.nextPage();
  const after = Number(vm.page || 0);
  return after > before;
}
"""


def build_playwright_extraction_config(*, keyword: str, max_pages: int) -> PlaywrightExtractionConfig:
    return PlaywrightExtractionConfig(
        profiles=(
            PlaywrightProfileConfig(
                profile="paginated_eval",
                script_args={"keyword": keyword},
                seed_script=PLANTPAD_SEED_SCRIPT,
                extract_script=PLANTPAD_EXTRACT_SCRIPT,
                next_page_script=PLANTPAD_NEXT_PAGE_SCRIPT,
                max_pages=max_pages,
                wait_after_action_ms=700,
                source_tag="vue-search",
                source_classes=("table-button",),
            ),
        )
    )


def print_navigation_state_summary(docs: list) -> None:
    nav_docs = [doc for doc in docs if isinstance(getattr(doc, "metadata", None), dict) and "web_navigation_state_index" in doc.metadata]
    if not nav_docs:
        print("Navigation states: 0")
        return

    print(f"Navigation states: {len(nav_docs)}")
    for doc in nav_docs[:5]:
        metadata = doc.metadata
        print(
            "  "
            f"state={metadata.get('web_navigation_state_index')}/"
            f"{metadata.get('web_navigation_state_count')} "
            f"click={metadata.get('web_navigation_click_count')} "
            f"source={metadata.get('source')}"
        )


def run_sync_example(
    *,
    fetch_mode: str,
    playwright_visible: bool,
    keyword: str,
    max_result_pages: int,
    use_custom_pagination_script: bool,
) -> None:
    print_section("17A1. Sync WebLoader (plantpad)")
    print(
        f"fetch_mode={fetch_mode} playwright_visible={playwright_visible} "
        f"use_custom_pagination_script={use_custom_pagination_script}"
    )
    loader = WebLoader(
        url=TARGET_URL,
        depth=3,
        output_format="markdown",
        fetch_mode=fetch_mode,
        crawl_scope="same_host",
        follow_download_links=False,
        cleanup_config=CLEANUP_CONFIG,
        playwright_visible=playwright_visible,
        playwright_navigation_config=PlaywrightNavigationConfig(
            enabled=True,
            max_clicks=max_result_pages,
            max_states=max_result_pages + 1,
        ),
        playwright_extraction_config=(
            build_playwright_extraction_config(keyword=keyword, max_pages=max_result_pages)
            if use_custom_pagination_script
            else None
        ),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    print_navigation_state_summary(docs)
    save_json_results(docs, "17A_web_loader_plantpad", "sync_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17A_web_loader_plantpad", "sync_errors")


async def run_async_example(
    *,
    fetch_mode: str,
    playwright_visible: bool,
    keyword: str,
    max_result_pages: int,
    use_custom_pagination_script: bool,
) -> None:
    print_section("17A2. Async WebLoader (plantpad)")
    print(
        f"fetch_mode={fetch_mode} playwright_visible={playwright_visible} "
        f"use_custom_pagination_script={use_custom_pagination_script}"
    )
    loader = AsyncWebLoader(
        url=TARGET_URL,
        depth=3,
        output_format="markdown",
        fetch_mode=fetch_mode,
        crawl_scope="same_host",
        follow_download_links=False,
        max_concurrency=4,
        cleanup_config=CLEANUP_CONFIG,
        playwright_visible=playwright_visible,
        playwright_navigation_config=PlaywrightNavigationConfig(
            enabled=True,
            max_clicks=max_result_pages,
            max_states=max_result_pages + 1,
        ),
        playwright_extraction_config=(
            build_playwright_extraction_config(keyword=keyword, max_pages=max_result_pages)
            if use_custom_pagination_script
            else None
        ),
    )
    docs = await loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    print_navigation_state_summary(docs)
    save_json_results(docs, "17A_web_loader_plantpad", "async_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17A_web_loader_plantpad", "async_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebLoader Plantpad demo with Playwright pagination extraction.")
    parser.add_argument("--requests-fallback", action="store_true", help="Use requests first and fallback to Playwright.")
    parser.add_argument("--playwright-visible", action="store_true", default=True, help="Run Playwright in visible (headed) mode.")
    parser.add_argument(
        "--use-custom-pagination-script",
        action="store_true",
        default=True,
        help="Use Plantpad-specific paginated_eval extractor (legacy path).",
    )
    parser.add_argument("--keyword", default=PLANTPAD_SEARCH_KEYWORD, help="Keyword used for search seeding.")
    parser.add_argument(
        "--max-result-pages",
        type=int,
        default=PLANTPAD_MAX_RESULT_PAGES,
        help="Maximum number of search result pages to extract from the Plantpad Vue table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_mode = "requests_fallback_playwright" if args.requests_fallback else "playwright"
    max_result_pages = max(1, args.max_result_pages)

    setup_environment()
    run_sync_example(
        fetch_mode=fetch_mode,
        playwright_visible=args.playwright_visible,
        keyword=args.keyword,
        max_result_pages=max_result_pages,
        use_custom_pagination_script=args.use_custom_pagination_script,
    )
    asyncio.run(
        run_async_example(
            fetch_mode=fetch_mode,
            playwright_visible=args.playwright_visible,
            keyword=args.keyword,
            max_result_pages=max_result_pages,
            use_custom_pagination_script=args.use_custom_pagination_script,
        )
    )


if __name__ == "__main__":
    main()
