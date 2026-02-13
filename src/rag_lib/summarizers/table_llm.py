from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from rag_lib.summarizers.table import TableSummarizer

from rag_lib.config import Settings

class LLMTableSummarizer(TableSummarizer):
    """
    Summarizes tables using a Language Model.
    """
    def __init__(self, llm: BaseChatModel, prompt_template: Optional[str] = None):
        self.llm = llm
        
        # Load default from settings if not provided
        if prompt_template is None:
            settings = Settings()
            prompt_template = settings.prompts.table_summarizer_template
            
        self.prompt = ChatPromptTemplate.from_template(prompt_template)

    def summarize(self, markdown_table: str) -> str:
        """
        Generates a summary for the given markdown table.
        """
        chain = self.prompt | self.llm
        response = chain.invoke({"table_content": markdown_table})
        
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
        response = await chain.ainvoke({"table_content": markdown_table})
        
        content = response.content
        if isinstance(content, list):
            return "".join([block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text'])
        
        return str(content)
