from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_lib.core.logger import logger

class ClusterSummarizer:
    """
    Summarizes a cluster of texts using an LLM.
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # Prompt from RAPTOR paper/reference
        self.template = """Here is a sub-set of documents.
        
        Give a detailed summary of the documentation provided.
        
        Documentation:
        {context}
        """ # TODO: Make this customizable?
        
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize(self, texts: List[str]) -> str:
        """
        Summarizes a list of texts into a single summary.
        """
        context = "\n\n---\n\n".join(texts)
        try:
            # TODO: Handle context length limits here if needed (e.g. map-reduce)
            summary = self.chain.invoke({"context": context})
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize cluster: {e}")
            return "Summary generation failed."

    async def asummarize(self, texts: List[str]) -> str:
        """
        Async version of summarize.
        """
        context = "\n\n---\n\n".join(texts)
        try:
            summary = await self.chain.ainvoke({"context": context})
            return summary
        except Exception as e:
            logger.error(f"Failed to summarize cluster (async): {e}")
            return "Summary generation failed."
