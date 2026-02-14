from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_lib.graph.store import BaseGraphStore, NetworkXGraphStore
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.logger import logger

class CommunitySummarizer:
    """
    Generates summaries for graph communities.
    """
    def __init__(self, llm: BaseChatModel, store: BaseGraphStore):
        self.llm = llm
        self.store = store
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant responsible for summarizing communities of entities in a knowledge graph.
            
            Community Entities and Descriptions:
            {context}
            
            Please provide a comprehensive summary of this community, highlighting the common themes and relationships.
            Start with a high-level overview.
            """
        )
        self.output_parser = StrOutputParser()

    def summarize(self, communities: Dict[int, List[str]]) -> List[Segment]:
        """
        Generates summaries for the provided communities.
        Returns a list of Segments representing these summaries (for indexing).
        """
        summaries = []
        logger.info(f"Generating summaries for {len(communities)} communities...")
        
        for cid, node_ids in communities.items():
            try:
                summary_text = self._summarize_single_community(node_ids)
                if summary_text:
                    # Create a Segment for this summary
                    seg = Segment(
                        content=summary_text,
                        type=SegmentType.TEXT, # Or add a new type?
                        metadata={
                            "community_id": cid, 
                            "node_ids": node_ids, 
                            "is_community_summary": True,
                            "type": "community_summary" # Explicit metadata type
                        }
                    )
                    summaries.append(seg)
            except Exception as e:
                logger.error(f"Failed to summarize community {cid}: {e}")
                
        return summaries

    def _summarize_single_community(self, node_ids: List[str]) -> str:
        # 1. Gather context
        context_parts = []
        for nid in node_ids:
            node = self.store.get_node(nid)
            if node:
                desc = node.description or "No description"
                context_parts.append(f"- Name: {node.label}, Type: {node.type}, Description: {desc}")
        
        if not context_parts:
            return ""
            
        context = "\n".join(context_parts)
        
        # 2. Invoke LLM
        chain = self.prompt | self.llm | self.output_parser
        return chain.invoke({"context": context})
