import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.loaders import AsyncWebLoader, WebCleanupConfig, WebLoader


TARGET_URL = "https://quotes.toscrape.com"

CLEANUP_CONFIG = WebCleanupConfig(
    # Keep duplicate tag cleanup conservative for quotes.toscrape:
    # broad dedupe on div/p can strip the page's main text blocks.
    duplicate_tags=(),
    non_recursive_classes=("tag",),
    navigation_classes=("side_categories", "pager"),
    ignored_classes=(
        "footer",
        "row header-box",
        "breadcrumb",
        "header container-fluid",
        "icon-star",
        "image_container",
    ),
)


def run_sync_example() -> None:
    print_section("17B1. Sync WebLoader (quotes)")
    loader = WebLoader(
        url=TARGET_URL,
        depth=3,
        output_format="markdown",
        fetch_mode="requests",
        crawl_scope="same_host",
        follow_download_links=False,
        cleanup_config=CLEANUP_CONFIG,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17B_web_loader_quotes", "sync_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17B_web_loader_quotes", "sync_errors")


async def run_async_example() -> None:
    print_section("17B2. Async WebLoader (quotes)")
    loader = AsyncWebLoader(
        url=TARGET_URL,
        depth=3,
        output_format="markdown",
        fetch_mode="requests_fallback_playwright",
        crawl_scope="same_host",
        follow_download_links=False,
        max_concurrency=4,
        cleanup_config=CLEANUP_CONFIG,
    )
    docs = await loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17B_web_loader_quotes", "async_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17B_web_loader_quotes", "async_errors")


def main() -> None:
    setup_environment()
    run_sync_example()
    asyncio.run(run_async_example())


if __name__ == "__main__":
    main()
