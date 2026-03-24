from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from rag_lib import ImageLoader as RootImageLoader
from rag_lib.loaders import ImageLoader as PackageImageLoader
from rag_lib.loaders.image import ImageLoader


def _write_image(path: Path, *, image_format: str = "PNG", color: tuple[int, int, int] = (255, 255, 255)) -> Path:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path, format=image_format)
    return path


def _fake_tesseract_success(ocr_text: str, *, input_validator=None):
    def _run(args, capture_output, text, encoding, errors, check):
        if input_validator is not None:
            input_validator(Path(args[1]))
        output_path = Path(args[2]).with_suffix(".txt")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(ocr_text)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return _run


def test_image_loader_is_exported_from_public_modules() -> None:
    assert RootImageLoader is ImageLoader
    assert PackageImageLoader is ImageLoader


def test_image_loader_renders_markdown_and_metadata(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.png", image_format="PNG")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="Concise image summary.")

    with patch("rag_lib.loaders.image.shutil.which", return_value="tesseract"), patch(
        "rag_lib.loaders.image.subprocess.run",
        side_effect=_fake_tesseract_success("Line 1\r\nLine 2\r\n\x0c"),
    ):
        docs = ImageLoader(str(image_path), llm=fake_llm, ocr_lang="eng").load()

    assert len(docs) == 1
    doc = docs[0]
    assert (
        doc.page_content
        == "# sample\n\n## Summary\n\nConcise image summary.\n\n## OCR\n\nLine 1\nLine 2"
    )
    assert doc.metadata["source"] == str(image_path)
    assert doc.metadata["source_type"] == "image"
    assert doc.metadata["output_format"] == "markdown"
    assert doc.metadata["image_format"] == "PNG"
    assert doc.metadata["mime_type"] == "image/png"
    assert doc.metadata["ocr_engine"] == "tesseract"
    assert doc.metadata["ocr_lang"] == "eng"


def test_image_loader_uses_first_frame_for_multiframe_inputs(tmp_path: Path) -> None:
    image_path = tmp_path / "animated.gif"
    first = Image.new("RGB", (6, 6), color=(255, 0, 0))
    second = Image.new("RGB", (6, 6), color=(0, 0, 255))
    first.save(image_path, save_all=True, append_images=[second], format="GIF", loop=0)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="summary")

    def _assert_first_frame(input_path: Path) -> None:
        with Image.open(input_path) as normalized:
            assert normalized.getpixel((0, 0)) == (255, 0, 0)

    with patch("rag_lib.loaders.image.shutil.which", return_value="tesseract"), patch(
        "rag_lib.loaders.image.subprocess.run",
        side_effect=_fake_tesseract_success("gif text", input_validator=_assert_first_frame),
    ):
        docs = ImageLoader(str(image_path), llm=fake_llm).load()

    assert len(docs) == 1
    assert docs[0].metadata["image_format"] == "GIF"


def test_image_loader_emits_no_text_detected_when_ocr_is_empty(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "blank.png", image_format="PNG")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="blank summary")

    with patch("rag_lib.loaders.image.shutil.which", return_value="tesseract"), patch(
        "rag_lib.loaders.image.subprocess.run",
        side_effect=_fake_tesseract_success("\x0c"),
    ):
        docs = ImageLoader(str(image_path), llm=fake_llm).load()

    assert len(docs) == 1
    assert docs[0].page_content.endswith("## OCR\n\n(no text detected)")


def test_image_loader_raises_for_unreadable_image(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported or unreadable image"):
        ImageLoader(str(image_path), llm=MagicMock()).load()


def test_image_loader_raises_when_tesseract_is_missing(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.png", image_format="PNG")

    with patch("rag_lib.loaders.image.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="Tesseract executable not found"):
            ImageLoader(str(image_path), llm=MagicMock()).load()


def test_image_loader_raises_when_tesseract_fails(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.png", image_format="PNG")
    fake_llm = MagicMock()

    with patch("rag_lib.loaders.image.shutil.which", return_value="tesseract"), patch(
        "rag_lib.loaders.image.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="ocr failed"),
    ):
        with pytest.raises(RuntimeError, match="Tesseract OCR failed: ocr failed"):
            ImageLoader(str(image_path), llm=fake_llm).load()


def test_image_loader_passes_multimodal_prompt_with_ocr_context(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "document.jpg", image_format="JPEG")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="image summary")

    with patch("rag_lib.loaders.image.shutil.which", return_value="tesseract"), patch(
        "rag_lib.loaders.image.subprocess.run",
        side_effect=_fake_tesseract_success("Detected heading\nDetected body"),
    ):
        docs = ImageLoader(str(image_path), llm=fake_llm).load()

    assert len(docs) == 1
    call_args = fake_llm.invoke.call_args[0][0]
    assert len(call_args) == 1
    message = call_args[0]
    assert isinstance(message.content, list)
    assert message.content[0]["type"] == "text"
    assert "Detected heading\nDetected body" in message.content[0]["text"]
    assert message.content[1]["type"] == "image_url"
    assert str(message.content[1]["image_url"]["url"]).startswith("data:image/png;base64,")


def test_image_loader_uses_explicit_mini_mode_for_default_summary_llm(tmp_path: Path) -> None:
    image_path = _write_image(tmp_path / "sample.png", image_format="PNG")
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="summary")

    with patch("rag_lib.loaders.image.create_llm", return_value=fake_llm) as mock_create_llm, patch(
        "rag_lib.loaders.image.shutil.which",
        return_value="tesseract",
    ), patch(
        "rag_lib.loaders.image.subprocess.run",
        side_effect=_fake_tesseract_success("Detected text"),
    ):
        docs = ImageLoader(str(image_path)).load()

    assert len(docs) == 1
    mock_create_llm.assert_called_once_with(model_name="mini", streaming=False)
