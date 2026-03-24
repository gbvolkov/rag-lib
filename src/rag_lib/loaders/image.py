from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Formatter
from typing import List, Optional

from PIL import Image, ImageOps, UnidentifiedImageError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rag_lib.chunkers.language import detect_nltk_language
from rag_lib.config import Settings
from rag_lib.core.domain import Document
from rag_lib.core.logger import logger
from rag_lib.llm.factory import create_llm

SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF", "GIF"}
_DEFAULT_SUMMARY_GUIDANCE = (
    "\n\nAdditional requirements:\n"
    "- Write the summary in {target_language}.\n"
    "- Keep it concise and factual.\n"
    "- Aim to keep the summary within about {soft_max_chars} characters "
    "(soft guidance, not a strict limit).\n"
)


@dataclass(frozen=True)
class _NormalizedImage:
    image_format: str
    source_mime_type: str
    png_bytes: bytes


class _LLMImageSummarizer:
    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        prompt_template: str | None = None,
        soft_max_chars: int | None = None,
    ) -> None:
        settings = Settings()
        self.llm = llm or create_llm(model_name="mini", streaming=False)

        if prompt_template is None:
            prompt_template = settings.prompts.image_loader_summary_template
        if soft_max_chars is None:
            soft_max_chars = settings.prompts.image_loader_summary_soft_max_chars
        if soft_max_chars <= 0:
            raise ValueError("soft_max_chars must be > 0.")

        self.soft_max_chars = soft_max_chars
        self.template = self._with_summary_guidance(prompt_template)

    def summarize(self, *, image_name: str, image_bytes: bytes, ocr_text: str) -> str:
        payload = self._build_payload(image_name=image_name, ocr_text=ocr_text)
        prompt_text = self.template.format(**payload)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": self._to_data_url(image_bytes, "image/png")}},
            ]
        )
        response = self.llm.invoke([message])
        summary = self._response_to_text(response.content)
        if not summary:
            raise RuntimeError("Image summary LLM returned empty content.")
        return summary

    def _build_payload(self, *, image_name: str, ocr_text: str) -> dict[str, object]:
        normalized_ocr = ocr_text if ocr_text and ocr_text.strip() else "(no text detected)"
        return {
            "image_name": image_name,
            "ocr_text": normalized_ocr,
            "target_language": self._detect_target_language(ocr_text),
            "soft_max_chars": self.soft_max_chars,
        }

    @staticmethod
    def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _response_to_text(content: object) -> str:
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(str(block.get("text", "")))
            return "".join(texts).strip()
        return str(content).strip()

    @staticmethod
    def _detect_target_language(ocr_text: str) -> str:
        normalized = re.sub(r"[|:\-\n]+", " ", ocr_text or "").strip()
        if not normalized:
            return "english"
        return detect_nltk_language(normalized, default="english")

    def _with_summary_guidance(self, prompt_template: str) -> str:
        variables = self._extract_template_variables(prompt_template)
        if {"target_language", "soft_max_chars"}.issubset(variables):
            return prompt_template
        return f"{prompt_template.rstrip()}{_DEFAULT_SUMMARY_GUIDANCE}"

    @staticmethod
    def _extract_template_variables(template: str) -> set[str]:
        formatter = Formatter()
        variables: set[str] = set()
        try:
            for _, field_name, _, _ in formatter.parse(template):
                if not field_name:
                    continue
                normalized = field_name.split(".", 1)[0].split("[", 1)[0]
                variables.add(normalized)
        except ValueError:
            return set()
        return variables


class ImageLoader:
    """
    Strict image -> Markdown loader with Tesseract OCR and multimodal LLM summary.
    """

    def __init__(
        self,
        file_path: str,
        *,
        llm: BaseChatModel | None = None,
        ocr_lang: str | None = None,
        tesseract_cmd: str | None = None,
    ) -> None:
        self.file_path = file_path
        self.llm = llm
        self.ocr_lang = ocr_lang.strip() if ocr_lang else None
        self.tesseract_cmd = tesseract_cmd.strip() if tesseract_cmd else None

    def load(self) -> List[Document]:
        logger.info("Loading image: %s", self.file_path)

        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        normalized = self._load_normalized_image(path)
        ocr_text = self._run_ocr(normalized.png_bytes)
        summary = self._resolve_summarizer().summarize(
            image_name=path.name,
            image_bytes=normalized.png_bytes,
            ocr_text=ocr_text,
        )
        markdown = self._render_markdown(path.stem, summary, ocr_text)

        metadata = {
            "source": self.file_path,
            "source_type": "image",
            "output_format": "markdown",
            "image_format": normalized.image_format,
            "mime_type": normalized.source_mime_type,
            "ocr_engine": "tesseract",
        }
        if self.ocr_lang:
            metadata["ocr_lang"] = self.ocr_lang

        return [Document(page_content=markdown, metadata=metadata)]

    def _load_normalized_image(self, path: Path) -> _NormalizedImage:
        try:
            with Image.open(path) as image:
                image_format = (image.format or "").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                    raise ValueError(
                        f"Unsupported image format '{image_format or 'unknown'}'. Supported formats: {supported}."
                    )

                image.seek(0)
                frame = ImageOps.exif_transpose(image)
                normalized = frame.convert("RGB")
                buffer = BytesIO()
                normalized.save(buffer, format="PNG")

                source_mime_type = (
                    Image.MIME.get(image_format)
                    or mimetypes.guess_type(str(path))[0]
                    or "application/octet-stream"
                )

                return _NormalizedImage(
                    image_format=image_format,
                    source_mime_type=source_mime_type,
                    png_bytes=buffer.getvalue(),
                )
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unsupported or unreadable image: {self.file_path}") from exc

    def _run_ocr(self, png_bytes: bytes) -> str:
        command = self._resolve_tesseract_command()

        with tempfile.TemporaryDirectory(prefix="rag_lib_image_ocr_") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "input.png"
            output_base = temp_root / "ocr_output"
            input_path.write_bytes(png_bytes)

            args = [command, str(input_path), str(output_base)]
            if self.ocr_lang:
                args.extend(["-l", self.ocr_lang])

            try:
                completed = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Tesseract executable not found: {command}") from exc

            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"Tesseract OCR failed: {details or 'unknown error'}")

            output_path = output_base.with_suffix(".txt")
            if not output_path.exists():
                raise RuntimeError("Tesseract OCR completed but produced no text output.")

            raw_text = output_path.read_text(encoding="utf-8", errors="ignore")
            return self._normalize_ocr_text(raw_text)

    def _resolve_tesseract_command(self) -> str:
        if self.tesseract_cmd:
            return self.tesseract_cmd
        command = shutil.which("tesseract")
        if not command:
            raise FileNotFoundError("Tesseract executable not found on PATH.")
        return command

    def _resolve_summarizer(self) -> _LLMImageSummarizer:
        return _LLMImageSummarizer(llm=self.llm)

    @staticmethod
    def _normalize_ocr_text(raw_text: str) -> str:
        normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "").strip()
        return normalized

    @staticmethod
    def _render_markdown(title: str, summary: str, ocr_text: str) -> str:
        rendered_ocr = ocr_text if ocr_text else "(no text detected)"
        return f"# {title}\n\n## Summary\n\n{summary}\n\n## OCR\n\n{rendered_ocr}".strip()
