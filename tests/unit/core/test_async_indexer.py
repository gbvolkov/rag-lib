import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, Mock
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer

def test_aindex_calls_aadd_texts():
    async def _test():
        mock_store = MagicMock()
        mock_store.aadd_texts = AsyncMock()
        
        indexer = Indexer(vector_store=mock_store, embeddings=MagicMock())
        
        segments = [
            Segment(content="Content 1", type=SegmentType.TEXT, segment_id="1"),
            Segment(content="Content 2", type=SegmentType.TEXT, segment_id="2")
        ]
        
        await indexer.aindex(segments, batch_size=2)
        
        # Check if aadd_texts was awaited
        mock_store.aadd_texts.assert_awaited()
        
        # Check arguments
        call_args = mock_store.aadd_texts.await_args
        # call_args.kwargs['texts'] should be ["Content 1", "Content 2"]
        assert call_args.kwargs['texts'] == ["Content 1", "Content 2"]
        assert call_args.kwargs['ids'] == ["1", "2"]
        
    asyncio.run(_test())
