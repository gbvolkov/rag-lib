from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from rag_lib.core.domain import Segment
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.store import BaseGraphStore
from rag_lib.core.logger import logger
import json

class EntityExtractor:
    """
    Extracts entities and relationships from Segments using an LLM.
    Populates a GraphStore.
    """
    def __init__(self, llm: BaseChatModel, store: BaseGraphStore):
        self.llm = llm
        self.store = store
        
        # Prompt inspired by LightRAG for Entity/Relation extraction
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a knowledge graph extractor.
            Identify key entities (Person, Organization, Location, Event, Concept) and relationships from the text.
            
            Text:
            {text}
            
            Return a JSON object with:
            "entities": [{{"name": "Entity Name", "type": "Type", "description": "Short description"}}],
            "relationships": [{{"source": "Entity Name", "target": "Entity Name", "relation": "RELATION_TYPE", "weight": 1.0}}]
            
            Ensure output is valid JSON.
            """
        )
        self.parser = JsonOutputParser()

    def process_segments(self, segments: List[Segment]) -> None:
        logger.info(f"Extracting graph data from {len(segments)} segments...")
        for seg in segments:
            try:
                self._process_segment(seg)
            except Exception as e:
                logger.error(f"Failed to extract graph from segment {seg.segment_id}: {e}")

    def _process_segment(self, segment: Segment) -> None:
        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"text": segment.content})
        except Exception as e:
            logger.warning(f"LLM extraction failed for {segment.segment_id}, trying raw text fallback: {e}")
            return # Skip if parsing fails

        # Add Nodes
        for ent in result.get("entities", []):
            node = GraphNode(
                id=ent["name"], # Simple ID for now
                type=ent.get("type", "concept"),
                label=ent["name"],
                description=ent.get("description"),
                source_segment_id=segment.segment_id
            )
            self.store.add_node(node)
            
            # Link Segment -> Entity (Optional, for traceability)
            # self.store.add_edge(GraphEdge(source_id=segment.segment_id, target_id=ent["name"], relation_type="MENTIONS"))

        # Add Edges
        for rel in result.get("relationships", []):
            edge = GraphEdge(
                source_id=rel["source"],
                target_id=rel["target"],
                relation_type=rel.get("relation", "RELATED_TO"),
                weight=rel.get("weight", 1.0),
                source_segment_id=segment.segment_id
            )
            self.store.add_edge(edge)

    async def aprocess_segments(self, segments: List[Segment], concurrency: int = 5) -> None:
        """Async version with concurrency control"""
        import asyncio
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _bounded_process(seg):
            async with semaphore:
                try:
                    await self._aprocess_segment(seg)
                except Exception as e:
                    logger.error(f"Async graph extraction failed for {seg.segment_id}: {e}")

        tasks = [_bounded_process(seg) for seg in segments]
        await asyncio.gather(*tasks)

    async def _aprocess_segment(self, segment: Segment) -> None:
        chain = self.prompt | self.llm | self.parser
        result = await chain.ainvoke({"text": segment.content})
        
        for ent in result.get("entities", []):
            node = GraphNode(
                id=ent["name"],
                type=ent.get("type", "concept"),
                label=ent["name"],
                description=ent.get("description"),
                source_segment_id=segment.segment_id
            )
            await self.store.aadd_node(node)

        for rel in result.get("relationships", []):
            edge = GraphEdge(
                source_id=rel["source"],
                target_id=rel["target"],
                relation_type=rel.get("relation", "RELATED_TO"),
                weight=rel.get("weight", 1.0),
                source_segment_id=segment.segment_id
            )
            await self.store.aadd_edge(edge)
