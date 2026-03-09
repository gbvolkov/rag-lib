from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_lib.summarizers.presentation_visual import (
    LLMPresentationVisualSummarizer,
    PresentationVisual,
)


def test_presentation_visual_summarizer_uses_prompt_from_config() -> None:
    old_env = os.environ.copy()
    try:
        os.environ["PROMPT_PRESENTATION_VISUAL_SUMMARIZER_TEMPLATE"] = (
            "You are a slide analyst.\n"
            "Visual kind: {visual_kind}\n"
            "Summary:"
        )
        summarizer = LLMPresentationVisualSummarizer(llm=MagicMock())
        assert "slide analyst" in summarizer.template
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_presentation_visual_summarizer_attaches_image_payload() -> None:
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = SimpleNamespace(content="image summary")
    summarizer = LLMPresentationVisualSummarizer(llm=fake_llm, prompt_template="Summarize {visual_kind}: {structured_markdown}")

    visual = PresentationVisual(
        kind="image",
        slide_index=1,
        slide_title="Gallery",
        shape_name="Hero",
        mime_type="image/png",
        image_bytes=b"png-bytes",
        structured_markdown="Product preview",
    )

    summary = summarizer.summarize(visual)

    assert summary == "image summary"
    call_args = fake_llm.invoke.call_args[0][0]
    assert len(call_args) == 1
    message = call_args[0]
    assert isinstance(message.content, list)
    assert message.content[0]["type"] == "text"
    assert message.content[1]["type"] == "image_url"
    assert str(message.content[1]["image_url"]["url"]).startswith("data:image/png;base64,")
