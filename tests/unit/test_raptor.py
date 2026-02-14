import pytest
import numpy as np
from unittest.mock import MagicMock
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.raptor import RaptorProcessor
from langchain_community.chat_models import FakeListChatModel

class MockEmbeddings:
    def embed_documents(self, texts):
        # Return random vectors
        return [np.random.rand(10) for _ in texts]
    
    async def aembed_documents(self, texts):
        return [np.random.rand(10) for _ in texts]

@pytest.fixture
def mock_clustering():
    clustering = MagicMock()
    # Mock clustering: Level 1 -> 2 clusters (0, 1), Level 2 -> 1 cluster (0)
    # Side effect sequence for calls:
    # 1. Level 1 (input 4 segs) -> returns [0, 0, 1, 1] (2 clusters)
    # 2. Level 2 (input 2 summaries) -> returns [0, 0] (1 cluster)
    clustering.perform_clustering.side_effect = [
        [np.array([0]), np.array([0]), np.array([1]), np.array([1])], # Level 1
        [np.array([0]), np.array([0])] # Level 2
    ]
    return clustering

def test_raptor_processor_logic(mock_clustering):
    # Setup
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(responses=["Summary L1-C0", "Summary L1-C1", "Summary L2-C0"])
    
    processor = RaptorProcessor(llm=llm, embeddings=embeddings)
    processor.clustering = mock_clustering # Inject mock
    
    # Create 4 leaf segments
    leaves = [
        Segment(content=f"Leaf {i}", type=SegmentType.TEXT) 
        for i in range(4)
    ]
    
    # Execute
    results = processor.process_segments(leaves)
    
    # Verification
    # Expected: 
    # Level 0: 4 leaves
    # Level 1: 2 summaries (from 2 clusters of leaves)
    # Level 2: 1 summary (from clustering Level 1 summaries)
    # Total: 4 + 2 + 1 = 7 segments
    
    assert len(results) == 7
    
    # Check hierarchy
    summaries = [s for s in results if s.metadata.get("is_raptor_summary")]
    assert len(summaries) == 3
    
    l1_summaries = [s for s in summaries if s.metadata["raptor_level"] == 1]
    assert len(l1_summaries) == 2
    assert l1_summaries[0].content in ["Summary L1-C0", "Summary L1-C1"]
    
    l2_summaries = [s for s in summaries if s.metadata["raptor_level"] == 2]
    assert len(l2_summaries) == 1
    assert l2_summaries[0].content == "Summary L2-C0"
    
    # Verify Metadata (Children pointers)
    # L2 summary children should be L1 summaries
    child_ids = l2_summaries[0].metadata["raptor_child_ids"]
    assert len(child_ids) == 2
    assert set(child_ids) == set(s.segment_id for s in l1_summaries)

@pytest.mark.asyncio
async def test_async_raptor_processor(mock_clustering):
    # Setup
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(responses=["Summary L1-C0", "Summary L1-C1", "Summary L2-C0"])
    
    processor = RaptorProcessor(llm=llm, embeddings=embeddings)
    processor.clustering = mock_clustering # Inject mock
    
    leaves = [
        Segment(content=f"Leaf {i}", type=SegmentType.TEXT) 
        for i in range(4)
    ]
    
    results = await processor.aprocess_segments(leaves)
    
    assert len(results) == 7
    summaries = [s for s in results if s.metadata.get("is_raptor_summary")]
    assert len(summaries) == 3
