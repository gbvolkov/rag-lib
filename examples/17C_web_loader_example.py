import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.loaders import AsyncWebLoader, WebCleanupConfig, WebLoader


TARGET_URL = "https://example.com"

# Mirrors the sample profile from recursive_scrapper/utils/retriever.py
CLEANUP_CONFIG = WebCleanupConfig(
    duplicate_tags=("div", "p", "table"),
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


def run_sync_example(*, ignore_https_errors: bool) -> None:
    print_section("17C1. Sync WebLoader (example.com)")
    print(f"ignore_https_errors={ignore_https_errors}")
    loader = WebLoader(
        url=TARGET_URL,
        depth=2,
        output_format="markdown",
        fetch_mode="requests",
        crawl_scope="allow_all",
        follow_download_links=False,
        cleanup_config=CLEANUP_CONFIG,
        ignore_https_errors=ignore_https_errors,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17C_web_loader_example", "sync_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17C_web_loader_example", "sync_errors")


async def run_async_example(*, ignore_https_errors: bool) -> None:
    print_section("17C2. Async WebLoader (example.com)")
    print(f"ignore_https_errors={ignore_https_errors}")
    loader = AsyncWebLoader(
        url=TARGET_URL,
        depth=2,
        output_format="markdown",
        fetch_mode="requests",
        crawl_scope="allow_all",
        follow_download_links=False,
        max_concurrency=4,
        cleanup_config=CLEANUP_CONFIG,
        ignore_https_errors=ignore_https_errors,
    )
    docs = await loader.load()
    print(f"Loaded {len(docs)} documents.")
    print(f"Stats: {loader.last_stats}")
    print(f"Errors: {len(loader.last_errors)}")
    save_json_results(docs, "17C_web_loader_example", "async_documents")
    if loader.last_errors:
        save_json_results(loader.last_errors, "17C_web_loader_example", "async_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebLoader example.com demo")
    parser.add_argument(
        "--ignore-https-errors",
        action="store_true",
        default=True,
        help="Disable TLS certificate verification for requests/Playwright (insecure workaround).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_environment()
    run_sync_example(ignore_https_errors=args.ignore_https_errors)
    asyncio.run(run_async_example(ignore_https_errors=args.ignore_https_errors))


if __name__ == "__main__":
    main()
