import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from example_utils import print_section, save_json_results, setup_environment
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.image import ImageLoader

"""
E2E Example 18: Image Loader OCR + Summary

Features Tested:
1. ImageLoader over raster images from ./docs.
2. Tesseract OCR with Russian + English language hints.
3. Mini LLM multimodal summarization.
4. Saving only loaded_documents into docs/results.
"""


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _resolve_image_paths(docs_dir: Path) -> list[Path]:
    supported_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
    return sorted(
        [
            path
            for path in docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ],
        key=lambda path: path.name.lower(),
    )


def main() -> None:
    _configure_console_encoding()
    setup_environment()
    print_section("18. Image Loader OCR + Summary")

    docs_dir = Path(__file__).parent.parent / "docs"
    image_paths = _resolve_image_paths(docs_dir)
    if not image_paths:
        print(f"No supported image files found in: {docs_dir}")
        return

    print(f"Found {len(image_paths)} image(s) in {docs_dir}:")
    for image_path in image_paths:
        print(f"- {image_path.name}")

    llm = create_llm(provider="openai", model_name="mini", streaming=False)
    loaded_documents = []
    failed_images: list[str] = []

    print_section("1. Loading Images")
    for image_path in image_paths:
        print(f"Loading {image_path.name} using ImageLoader...")
        try:
            docs = ImageLoader(
                str(image_path),
                llm=llm,
                ocr_lang="rus+eng",
            ).load()
        except Exception as exc:
            failed_images.append(image_path.name)
            print(f"Failed to load {image_path.name}: {exc}")
            continue

        loaded_documents.extend(docs)
        doc = docs[0]
        print(f"Loaded {image_path.name}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {doc.page_content[:220]}...")

    if not loaded_documents:
        print("No documents were loaded successfully.")
        return

    print_section("2. Saving Results")
    print(f"Loaded {len(loaded_documents)} document(s) successfully.")
    if failed_images:
        print(f"Skipped {len(failed_images)} failed image(s): {', '.join(failed_images)}")

    save_json_results(loaded_documents, "18_image_loader", "loaded_documents")


if __name__ == "__main__":
    main()
