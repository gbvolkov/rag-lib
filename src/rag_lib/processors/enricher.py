from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.logger import logger
from rag_lib.config import settings

class SegmentEnricher:
    """
    Enriches segments with metadata using an LLM.
    - Title extraction
    - Keyword generation
    - One-line summary (if missing)
    """
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        # Define default template
        # We ask for a JSON-like structure or structured output.
        # For robustness without forceful JSON mode, we use a simple format.
        self.prompt = ChatPromptTemplate.from_template(
            "You are a metadata extraction system. Analyze the following text segment.\n"
            "Extract the following metadata directly:\n"
            "1. Title (A short, descriptive title for this segment)\n"
            "2. Keywords (Comma-separated list of top 5 keywords)\n"
            "3. Summary (A concise one-line summary of the content)\n\n"
            "Text Segment:\n{text}\n\n"
            "Output Format:\n"
            "Title: <title>\n"
            "Keywords: <keywords>\n"
            "Summary: <summary>"
        )

    def enrich(self, segments: List[Segment]) -> List[Segment]:
        logger.info(f"Enriching {len(segments)} segments...")
        enriched_segments = []
        
        for seg in segments:
            # Skip enrichment for very short segments or already enriched ones logic could go here
            if seg.type == SegmentType.TEXT and len(seg.content) > 50:
                try:
                    self._enrich_segment(seg)
                except Exception as e:
                    logger.error(f"Failed to enrich segment {seg.segment_id}: {e}")
            
            enriched_segments.append(seg)
            
        return enriched_segments

    def _enrich_segment(self, segment: Segment) -> None:
        chain = self.prompt | self.llm
        response = chain.invoke({"text": segment.content})
        
        content = response.content
        logger.debug(f"LLM Response Content: {content}")
        
        if isinstance(content, list):
             # Handle complex block types if any
             content = "".join([b['text'] for b in content if 'text' in b])
        
        content = str(content)
        
        # Parse output
        lines = content.strip().split('\n')
        title = "Untitled"
        keywords = []
        summary = ""
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                k_str = line.split(":", 1)[1].strip()
                keywords = [k.strip() for k in k_str.split(',') if k.strip()]
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
        
        # Update Metadata
        segment.metadata["generated_title"] = title
        segment.metadata["keywords"] = keywords
        segment.metadata["summary"] = summary
        
        # Inject metadata into content for better retrieval context
        rich_header = (
            f"Title: {title}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Summary: {summary}\n\n"
        )
        segment.content = rich_header + segment.content
        
        logger.debug(f"Enriched {segment.segment_id}: Title='{title}', Keywords={keywords}, Summary='{summary}'")

    async def aenrich(self, segments: List[Segment]) -> List[Segment]:
        """Async version of enrich"""
        logger.info(f"Enriching {len(segments)} segments asynchronously...")
        # TODO: Use asyncio.gather for parallelism
        # For now sequential async to be safe
        for seg in segments:
            if seg.type == SegmentType.TEXT and len(seg.content) > 50:
                try:
                     await self._aenrich_segment(seg)
                except Exception as e:
                    logger.error(f"Failed to enrich segment {seg.segment_id}: {e}")
        return segments

    async def _aenrich_segment(self, segment: Segment) -> None:
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"text": segment.content})
        
        content = str(response.content)
        lines = content.strip().split('\n')
        title = "Untitled"
        keywords = []
        summary = ""
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                k_str = line.split(":", 1)[1].strip()
                keywords = [k.strip() for k in k_str.split(',') if k.strip()]
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
        
        segment.metadata["generated_title"] = title
        segment.metadata["keywords"] = keywords
        segment.metadata["summary"] = summary
        
        # Inject metadata into content
        rich_header = (
            f"Title: {title}\n"
            f"Keywords: {', '.join(keywords)}\n"
            f"Summary: {summary}\n\n"
        )
        segment.content = rich_header + segment.content
        
        logger.debug(f"Enriched {segment.segment_id}: Title='{title}', Keywords={keywords}, Summary='{summary}'")
