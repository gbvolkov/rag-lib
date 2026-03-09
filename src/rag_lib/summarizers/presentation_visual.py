from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from string import Formatter
from typing import Literal, Optional, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from rag_lib.chunkers.language import detect_nltk_language
from rag_lib.config import Settings
from rag_lib.llm.factory import create_llm

_DEFAULT_SUMMARY_GUIDANCE = (
    "\n\nAdditional requirements:\n"
    "- Write the summary in {target_language}.\n"
    "- Keep it concise and factual.\n"
    "- Aim to keep the summary within about {soft_max_chars} characters "
    "(soft guidance, not a strict limit).\n"
)


@dataclass(frozen=True)
class PresentationVisual:
    kind: Literal["image", "chart", "smartart"]
    slide_index: int
    slide_title: str
    shape_name: str
    mime_type: str
    image_bytes: bytes | None = None
    structured_markdown: str = ""


class PresentationVisualSummarizer(Protocol):
    def summarize(self, visual: PresentationVisual) -> str:
        ...


class LLMPresentationVisualSummarizer(PresentationVisualSummarizer):
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        prompt_template: Optional[str] = None,
        soft_max_chars: Optional[int] = None,
    ):
        settings = Settings()
        self.llm = llm or create_llm()

        if prompt_template is None:
            prompt_template = settings.prompts.presentation_visual_summarizer_template
        if soft_max_chars is None:
            soft_max_chars = settings.prompts.presentation_visual_summarizer_soft_max_chars
        if soft_max_chars <= 0:
            raise ValueError("soft_max_chars must be > 0.")

        self.soft_max_chars = soft_max_chars
        self.template = self._with_summary_guidance(prompt_template)

    def summarize(self, visual: PresentationVisual) -> str:
        payload = self._build_payload(visual)
        prompt_text = self.template.format(**payload)
        message = HumanMessage(content=self._build_message_content(visual, prompt_text))
        response = self.llm.invoke([message])
        return self._response_to_text(response.content)

    def _build_payload(self, visual: PresentationVisual) -> dict[str, object]:
        signal = "\n".join(
            part
            for part in [visual.slide_title, visual.shape_name, visual.structured_markdown]
            if part and part.strip()
        )
        target_language = self._detect_target_language(signal)
        return {
            "visual_kind": visual.kind,
            "slide_title": visual.slide_title,
            "shape_name": visual.shape_name,
            "structured_markdown": visual.structured_markdown or "(none)",
            "target_language": target_language,
            "soft_max_chars": self.soft_max_chars,
        }

    def _build_message_content(self, visual: PresentationVisual, prompt_text: str) -> str | list[dict[str, object]]:
        if visual.kind != "image":
            return prompt_text
        if not visual.image_bytes:
            raise ValueError("image_bytes are required for image summarization.")

        mime_type = visual.mime_type or "image/png"
        data_url = self._to_data_url(visual.image_bytes, mime_type)
        return [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

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
    def _detect_target_language(text: str) -> str:
        normalized = re.sub(r"[|:\-\n]+", " ", text or "")
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
