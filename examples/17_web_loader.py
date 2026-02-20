import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.loaders.web import WebLoader
from rag_lib.loaders.web_async import AsyncWebLoader


def print_related_examples_note() -> None:
    print_section("17. Related Multi-Source Samples")
    print("This legacy sample keeps the original quotes.toscrape sync/async flow.")
    print("For source-specific multi-site demos, see:")
    print("- examples/17A_web_loader_plantpad.py")
    print("- examples/17B_web_loader_quotes.py")
    print("- examples/17C_web_loader_example.py")


def run_sync_example() -> None:
    print_section("17A. Sync WebLoader")
    loader = WebLoader(
        url="https://quotes.toscrape.com",
        depth=2,
        output_format="markdown",
        fetch_mode="requests",
        crawl_scope="same_host",
        follow_download_links=False,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17_web_loader", "sync_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17_web_loader", "sync_errors")


async def run_async_example() -> None:
    print_section("17B. Async WebLoader")
    loader = AsyncWebLoader(
        url="https://quotes.toscrape.com",
        depth=2,
        output_format="markdown",
        fetch_mode="requests_fallback_playwright",
        crawl_scope="same_host",
        follow_download_links=False,
        max_concurrency=4,
    )
    docs = await loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17_web_loader", "async_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17_web_loader", "async_errors")


def main() -> None:
    setup_environment()
    print_related_examples_note()
    run_sync_example()
    asyncio.run(run_async_example())


if __name__ == "__main__":
    main()
