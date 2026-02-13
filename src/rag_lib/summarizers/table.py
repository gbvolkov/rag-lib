from typing import Protocol, Optional

class TableSummarizer(Protocol):
    def summarize(self, markdown_table: str) -> str:
        """
        Generates a natural language summary of a markdown table.
        """
        ...

class MockTableSummarizer:
    """
    A placeholder summarizer for testing/offline use.
    returns a static string or simple heuristic.
    """
    def summarize(self, markdown_table: str) -> str:
        # Simple heuristic: extract headers to pretend we understood it
        try:
            lines = markdown_table.strip().split('\n')
            if len(lines) >= 1:
                headers = lines[0]
                return f"Table with headers: {headers}. Contains data rows."
        except:
            pass
        return "Table Summary Placeholder"

class LLMTableSummarizer:
    """
    Real implementation would use LangChain or similar.
    Left as a skeleton for future integration.
    """
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def summarize(self, markdown_table: str) -> str:
        if not self.llm_client:
            raise NotImplementedError("LLM client not configured")
        # prompt = f"Summarize this table:\n{markdown_table}"
        # return self.llm_client.predict(prompt)
        return "LLM Summary"
