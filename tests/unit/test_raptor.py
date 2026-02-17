import pytest
import numpy as np
from unittest.mock import MagicMock
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.raptor import RaptorProcessor
from rag_lib.raptor.summarization import DEFAULT_SUMMARY_PROMPT_TEMPLATE
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
    llm = FakeListChatModel(
        responses=[
            "Cluster one summary sentence for node zero.",
            "Cluster two summary sentence for node one.",
            "Top summary node.",
        ]
    )
    
    processor = RaptorProcessor(llm=llm, embeddings=embeddings)
    processor.clustering = mock_clustering  # Backward-compat assignment
    processor.builder.clustering = mock_clustering  # Effective clustering service
    
    # Create 4 leaf segments
    leaves = [
        Segment(
            content=(
                f"Leaf {i} contains detailed project experience, architecture decisions, "
                f"and implementation outcomes across multiple systems. " * 4
            ),
            segment_id=f"leaf_{i}",
            type=SegmentType.TEXT,
        )
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
    assert l1_summaries[0].content in [
        "Cluster one summary sentence for node zero.",
        "Cluster two summary sentence for node one.",
    ]
    
    l2_summaries = [s for s in summaries if s.metadata["raptor_level"] == 2]
    assert len(l2_summaries) == 1
    assert l2_summaries[0].content == "Top summary node."
    
    # Verify Metadata (Children pointers)
    # L2 summary children should be L1 summaries
    child_ids = l2_summaries[0].metadata["raptor_child_ids"]
    assert len(child_ids) == 2
    assert set(child_ids) == set(s.segment_id for s in l1_summaries)

    # Verify Parent Links
    l2_id = l2_summaries[0].segment_id
    for s in l1_summaries:
        assert s.parent_id == l2_id
        assert s.metadata["raptor_parent_ids"] == [l2_id]

    leaves_in_results = [s for s in results if not s.metadata.get("is_raptor_summary")]
    assert len(leaves_in_results) == 4
    for leaf in leaves_in_results:
        assert leaf.parent_id is not None
        assert len(leaf.metadata["raptor_parent_ids"]) == 1

    # Verify top-down depth levels:
    # root summary -> level 0, level-1 summaries -> level 1, leaves -> level 2
    assert l2_summaries[0].level == 0
    for s in l1_summaries:
        assert s.level == 1
    for leaf in leaves_in_results:
        assert leaf.level == 2

@pytest.mark.asyncio
async def test_async_raptor_processor(mock_clustering):
    # Setup
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(
        responses=[
            "Cluster one summary sentence for node zero.",
            "Cluster two summary sentence for node one.",
            "Top summary node.",
        ]
    )
    
    processor = RaptorProcessor(llm=llm, embeddings=embeddings)
    processor.clustering = mock_clustering  # Backward-compat assignment
    processor.builder.clustering = mock_clustering  # Effective clustering service
    
    leaves = [
        Segment(
            content=(
                f"Leaf {i} contains detailed project experience, architecture decisions, "
                f"and implementation outcomes across multiple systems. " * 4
            ),
            segment_id=f"leaf_{i}",
            type=SegmentType.TEXT,
        )
        for i in range(4)
    ]
    
    results = await processor.aprocess_segments(leaves)
    
    assert len(results) == 7
    summaries = [s for s in results if s.metadata.get("is_raptor_summary")]
    assert len(summaries) == 3

    # Basic hierarchy linkage should also exist in async flow.
    non_root = [s for s in results if s.metadata.get("raptor_parent_ids")]
    assert len(non_root) == 6  # 4 leaves + 2 level-1 summaries

    root_summaries = [s for s in summaries if not s.metadata.get("raptor_parent_ids")]
    assert len(root_summaries) == 1
    assert root_summaries[0].level == 0

def test_raptor_processor_uses_default_summary_prompt():
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(responses=["ok"])

    processor = RaptorProcessor(llm=llm, embeddings=embeddings)

    assert processor.summarizer.template == DEFAULT_SUMMARY_PROMPT_TEMPLATE

def test_raptor_processor_accepts_custom_summary_prompt(mock_clustering):
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(
        responses=[
            "Cluster one summary sentence for node zero.",
            "Cluster two summary sentence for node one.",
            "Top summary node.",
        ]
    )
    custom_template = (
        "Summarize the context in {target_language}. "
        "Limit to {max_chars} chars and target ratio {target_ratio}.\n"
        "Context:\n{context}"
    )

    processor = RaptorProcessor(
        llm=llm,
        embeddings=embeddings,
        summary_prompt_template=custom_template,
    )
    processor.clustering = mock_clustering
    processor.builder.clustering = mock_clustering

    leaves = [
        Segment(
            content=(
                f"Leaf {i} contains detailed project experience, architecture decisions, "
                f"and implementation outcomes across multiple systems. " * 4
            ),
            segment_id=f"leaf_{i}",
            type=SegmentType.TEXT,
        )
        for i in range(4)
    ]

    results = processor.process_segments(leaves)

    assert processor.summarizer.template == custom_template
    assert any(s.metadata.get("is_raptor_summary") for s in results)

def test_raptor_processor_rejects_invalid_custom_summary_prompt():
    embeddings = MockEmbeddings()
    llm = FakeListChatModel(responses=["ok"])

    invalid_template = "Summarize this context only:\n{context}"

    with pytest.raises(ValueError):
        RaptorProcessor(
            llm=llm,
            embeddings=embeddings,
            summary_prompt_template=invalid_template,
        )
