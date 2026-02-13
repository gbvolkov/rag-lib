import pytest
from unittest.mock import MagicMock, AsyncMock, Mock
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.enricher import SegmentEnricher

@pytest.fixture
def mock_llm():
    return MagicMock()

def test_enrich_sync(mock_llm):
    # Setup
    enricher = SegmentEnricher(mock_llm)
    
    # Mock the prompt and chain to avoid pipe issues
    mock_chain = MagicMock()
    
    # Prepare response
    msg = Mock()
    msg.content = "Title: Test Document Title\nKeywords: key1, key2, key3"
    
    mock_chain.invoke.return_value = msg
    
    # Mock the prompt's __or__ method to return our mock chain
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    enricher.prompt = mock_prompt
    
    long_content = "This is a test content that is definitely longer than fifty characters to ensure enrichment triggers."
    segments = [Segment(content=long_content, type=SegmentType.TEXT)]
    
    # Act
    enriched = enricher.enrich(segments)
    
    # Assert
    assert len(enriched) == 1
    assert enriched[0].metadata["generated_title"] == "Test Document Title"
    assert enriched[0].metadata["keywords"] == ["key1", "key2", "key3"]

@pytest.mark.asyncio
async def test_enrich_async(mock_llm):
    # Setup
    enricher = SegmentEnricher(mock_llm)
    
    # Mock chain
    mock_chain = MagicMock()
    msg = Mock()
    msg.content = "Title: Test Document Title\nKeywords: key1, key2, key3"
    
    # Async mock needs strictly async return
    mock_chain.ainvoke = AsyncMock(return_value=msg)
    
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    enricher.prompt = mock_prompt
    
    long_content = "This is a test content that is definitely longer than fifty characters to ensure enrichment triggers."
    segments = [Segment(content=long_content, type=SegmentType.TEXT)]
    
    # Act
    enriched = await enricher.aenrich(segments)
    
    # Assert
    assert len(enriched) == 1
    assert enriched[0].metadata["generated_title"] == "Test Document Title"
    assert enriched[0].metadata["keywords"] == ["key1", "key2", "key3"]
