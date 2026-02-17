from string import Formatter
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_lib.core.logger import logger


DEFAULT_SUMMARY_PROMPT_TEMPLATE = """You are summarizing a cluster of document fragments.
Write the summary in {target_language}.
Keep it concise and factual.
Maximum length: {max_chars} characters.
Target compression ratio: {target_ratio}.
Do not add follow-up questions.

Documentation:
{context}
"""

_REQUIRED_PROMPT_VARIABLES = {"context", "target_language", "max_chars", "target_ratio"}


def _extract_template_variables(template: str) -> set[str]:
    formatter = Formatter()
    variables: set[str] = set()
    try:
        for _, field_name, _, _ in formatter.parse(template):
            if not field_name:
                continue
            # Support placeholders like `field.attr` or `field[idx]`.
            normalized = field_name.split(".", 1)[0].split("[", 1)[0]
            variables.add(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid summary prompt template: {exc}") from exc
    return variables


class ClusterSummarizer:
    """
    Summarizes a cluster of texts using an LLM.
    """
    def __init__(
        self,
        llm: BaseChatModel,
        summary_prompt_template: str | None = None,
    ):
        self.llm = llm

        self.template = summary_prompt_template or DEFAULT_SUMMARY_PROMPT_TEMPLATE
        self._validate_template(self.template)
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _validate_template(self, template: str) -> None:
        variables = _extract_template_variables(template)
        missing = _REQUIRED_PROMPT_VARIABLES - variables
        if missing:
            required = ", ".join(sorted(_REQUIRED_PROMPT_VARIABLES))
            missing_str = ", ".join(sorted(missing))
            raise ValueError(
                f"Invalid summary prompt template. Missing required placeholders: "
                f"{missing_str}. Required placeholders: {required}."
            )

    def summarize(
        self,
        texts: List[str],
        *,
        target_language: str = "english",
        max_chars: int = 1200,
        target_ratio: float = 0.35,
    ) -> str:
        """
        Summarizes a list of texts into a single summary.
        """
        context = "\n\n---\n\n".join(texts)
        payload = {
            "context": context,
            "target_language": target_language,
            "max_chars": max_chars,
            "target_ratio": target_ratio,
        }
        try:
            summary = self.chain.invoke(payload)
        except Exception as exc:
            logger.error(f"Failed to summarize cluster: {exc}")
            raise

        if not isinstance(summary, str):
            raise TypeError("Cluster summary must be a string.")
        return summary.strip()

    async def asummarize(
        self,
        texts: List[str],
        *,
        target_language: str = "english",
        max_chars: int = 1200,
        target_ratio: float = 0.35,
    ) -> str:
        """
        Async version of summarize.
        """
        context = "\n\n---\n\n".join(texts)
        payload = {
            "context": context,
            "target_language": target_language,
            "max_chars": max_chars,
            "target_ratio": target_ratio,
        }
        try:
            summary = await self.chain.ainvoke(payload)
        except Exception as exc:
            logger.error(f"Failed to summarize cluster (async): {exc}")
            raise

        if not isinstance(summary, str):
            raise TypeError("Cluster summary must be a string.")
        return summary.strip()
