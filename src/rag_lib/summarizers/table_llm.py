from string import Formatter
from typing import Optional
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from rag_lib.summarizers.table import TableSummarizer

from rag_lib.chunkers.language import detect_nltk_language
from rag_lib.config import Settings

_DEFAULT_SUMMARY_GUIDANCE = (
    "\n\nAdditional requirements:\n"
    "- Write the summary in {target_language}.\n"
    "- Keep it concise and factual.\n"
    "- Aim to keep the summary within about {soft_max_chars} characters "
    "(soft guidance, not a strict limit).\n"
)

class LLMTableSummarizer(TableSummarizer):
    """
    Summarizes tables using a Language Model.
    """
    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: Optional[str] = None,
        soft_max_chars: Optional[int] = None,
    ):
        self.llm = llm
        
        # Load default from settings if not provided
        settings = Settings()
        if prompt_template is None:
            prompt_template = settings.prompts.table_summarizer_template

        if soft_max_chars is None:
            soft_max_chars = settings.prompts.table_summarizer_soft_max_chars

        if soft_max_chars <= 0:
            raise ValueError("soft_max_chars must be > 0.")

        self.soft_max_chars = soft_max_chars
        self.prompt = ChatPromptTemplate.from_template(
            self._with_summary_guidance(prompt_template)
        )

    def summarize(self, markdown_table: str) -> str:
        """
        Generates a summary for the given markdown table.
        """
        chain = self.prompt | self.llm
        response = chain.invoke(self._build_payload(markdown_table))
        
        content = response.content
        if isinstance(content, list):
            # Handle list of content blocks (e.g. [{'type': 'text', 'text': '...'}])
            return "".join([block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text'])
        
        return str(content)

    async def asummarize(self, markdown_table: str) -> str:
        """
        Async version of summarize.
        """
        chain = self.prompt | self.llm
        response = await chain.ainvoke(self._build_payload(markdown_table))
        
        content = response.content
        if isinstance(content, list):
            return "".join([block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text'])
        
        return str(content)

    def _build_payload(self, markdown_table: str) -> dict[str, object]:
        target_language = self._detect_target_language(markdown_table)
        return {
            "table_content": markdown_table,
            "target_language": target_language,
            "soft_max_chars": self.soft_max_chars,
        }

    def _detect_target_language(self, markdown_table: str) -> str:
        # Drop markdown separators to improve language detection signal.
        normalized = re.sub(r"[|:\-\n]+", " ", markdown_table)
        detected = detect_nltk_language(normalized, default="english")
        return detected

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
