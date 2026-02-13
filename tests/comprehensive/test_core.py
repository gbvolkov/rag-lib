import pytest
from unittest.mock import MagicMock, AsyncMock, patch, Mock
import asyncio
from rag_lib.core.indexer import Indexer
from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.core.domain import Segment, SegmentType

# --- SegmentEnricher Tests ---

def test_enricher_robustness_garbage_output():
    # Scenario: LLM returns garbage that doesn't follow "Title: ... Keywords: ..." format
    mock_llm = MagicMock()
    msg = Mock()
    msg.content = "I am a large language model and I cannot help you."
    
    # Mock chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = msg
    
    mock_prompt = MagicMock()
    mock_prompt.__or__.return_value = mock_chain
    
    enricher = SegmentEnricher(mock_llm)
    enricher.prompt = mock_prompt
    
    long_content = "This content is definitely long enough to trigger the enrichment logic in the processor." * 2
    seg = Segment(content=long_content, type=SegmentType.TEXT)
    enriched_segs = enricher.enrich([seg])
    
    # Assertions
    # Should not crash.
    # Should have default or empty metadata.
    assert enriched_segs[0].metadata.get("generated_title") == "Untitled"
    assert enriched_segs[0].metadata.get("keywords") == []

def test_enricher_filters_short_segments():
    mock_llm = MagicMock()
    enricher = SegmentEnricher(mock_llm)
    # We don't even need to mock prompt/chain because it shouldn't be called.
    # To prove it, let's keep prompt unpatched/default (which would fail if called with mock LLM incorrectly)
    
    short_seg = Segment(content="Short.", type=SegmentType.TEXT)
    enriched_segs = enricher.enrich([short_seg])
    
    # Should be untouched
    assert "generated_title" not in enriched_segs[0].metadata

# --- Indexer Tests ---

def test_indexer_batching_logic():
    mock_store = MagicMock()
    mock_embeddings = MagicMock()
    
    indexer = Indexer(vector_store=mock_store, embeddings=mock_embeddings)
    
    # Create 15 segments
    segments = [Segment(content=f"Seg {i}", segment_id=str(i)) for i in range(15)]
    
    # Index with batch_size=5 -> Should call add_texts 3 times
    indexer.index(segments, batch_size=5)
    
    assert mock_store.add_texts.call_count == 3
    # Check first call args
    call_args = mock_store.add_texts.call_args_list[0]
    # args, kwargs. Kwargs: texts, metadatas, ids.
    assert len(call_args.kwargs['ids']) == 5

def test_indexer_parent_child_logic():
    mock_store = MagicMock()
    mock_embeddings = MagicMock()
    indexer = Indexer(vector_store=mock_store, embeddings=mock_embeddings)
    
    # Case 1: No Summary -> Embed Content
    seg1 = Segment(content="Raw Content", segment_id="1")
    
    # Case 2: Summary Present -> Embed Summary
    seg2 = Segment(content="Detailed Content", segment_id="2", metadata={"summary": "Short Summary"})
    
    indexer.index([seg1, seg2], batch_size=10)
    
    call_args = mock_store.add_texts.call_args[1] # kwargs
    texts = call_args['texts']
    metadatas = call_args['metadatas']
    
    # Check what was sent to be embedded
    assert texts[0] == "Raw Content"
    assert texts[1] == "Short Summary" # Key behavior check
    
    # Check payloads (must contain full content)
    assert metadatas[0]["content"] == "Raw Content"
    assert metadatas[0]["is_summary_embedding"] is False
    
    assert metadatas[1]["content"] == "Detailed Content"
    assert metadatas[1]["is_summary_embedding"] is True

@pytest.mark.asyncio
async def test_aindexer_integration():
    mock_store = MagicMock()
    mock_store.aadd_texts = AsyncMock() # Important!
    
    mock_embeddings = MagicMock()
    
    indexer = Indexer(vector_store=mock_store, embeddings=mock_embeddings)
    segments = [Segment(content="Async Test", segment_id="a1")]
    
    await indexer.aindex(segments)
    
    assert mock_store.aadd_texts.called
    assert mock_store.aadd_texts.await_count == 1
