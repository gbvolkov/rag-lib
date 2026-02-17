import numpy as np
import pytest
from unittest.mock import MagicMock

from langchain_community.chat_models import FakeListChatModel

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.raptor.summarization import ClusterSummarizer
from rag_lib.raptor.tree_builder import TreeBuilder


class MockEmbeddings:
    def embed_documents(self, texts):
        return [np.random.rand(10) for _ in texts]


def test_cluster_summarizer_rejects_missing_required_prompt_variables():
    llm = FakeListChatModel(responses=["ok"])
    invalid_template = "Summarize:\n{context}"

    with pytest.raises(ValueError):
        ClusterSummarizer(llm=llm, summary_prompt_template=invalid_template)


def test_cluster_summarizer_accepts_valid_custom_prompt_template():
    llm = FakeListChatModel(responses=["custom summary"])
    valid_template = (
        "Language={target_language}; max={max_chars}; ratio={target_ratio}\n"
        "{context}"
    )
    summarizer = ClusterSummarizer(llm=llm, summary_prompt_template=valid_template)

    result = summarizer.summarize(
        ["alpha text", "beta text"],
        target_language="english",
        max_chars=200,
        target_ratio=0.3,
    )

    assert summarizer.template == valid_template
    assert result == "custom summary"


def test_tree_builder_continues_when_summary_exceeds_budget():
    llm = FakeListChatModel(
        responses=[
            "This summary is deliberately too long for the strict per-cluster budget."
        ]
    )
    valid_template = (
        "Summarize in {target_language}. max={max_chars} ratio={target_ratio}\n"
        "{context}"
    )
    summarizer = ClusterSummarizer(llm=llm, summary_prompt_template=valid_template)

    clustering = MagicMock()
    clustering.perform_clustering.return_value = [np.array([0]), np.array([0])]

    builder = TreeBuilder(
        clustering_service=clustering,
        summarizer=summarizer,
        embeddings_model=MockEmbeddings(),
        summary_target_ratio=0.1,
        summary_max_chars=100,
        summary_min_chars=1,
        strict_quality=True,
    )

    leaves = [
        Segment(
            content="Leaf one includes comprehensive architecture and delivery details.",
            segment_id="leaf_1",
            type=SegmentType.TEXT,
        ),
        Segment(
            content="Leaf two includes additional implementation outcomes and constraints.",
            segment_id="leaf_2",
            type=SegmentType.TEXT,
        ),
    ]

    results = builder.build(leaves, n_levels=1)

    summaries = [s for s in results if s.metadata.get("is_raptor_summary")]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.metadata["raptor_summary_exceeds_max_chars"] is True
    assert summary.metadata["raptor_summary_chars"] > summary.metadata["raptor_summary_max_chars"]


def test_tree_builder_still_fails_on_language_mismatch_in_strict_mode():
    llm = FakeListChatModel(responses=["Это короткое резюме не на нужном языке."])
    valid_template = (
        "Summarize in {target_language}. max={max_chars} ratio={target_ratio}\n"
        "{context}"
    )
    summarizer = ClusterSummarizer(llm=llm, summary_prompt_template=valid_template)

    clustering = MagicMock()
    clustering.perform_clustering.return_value = [np.array([0]), np.array([0])]

    builder = TreeBuilder(
        clustering_service=clustering,
        summarizer=summarizer,
        embeddings_model=MockEmbeddings(),
        summary_target_ratio=0.8,
        summary_max_chars=200,
        summary_min_chars=1,
        strict_quality=True,
    )

    leaves = [
        Segment(
            content="Leaf one includes comprehensive architecture and delivery details.",
            segment_id="leaf_1",
            type=SegmentType.TEXT,
        ),
        Segment(
            content="Leaf two includes additional implementation outcomes and constraints.",
            segment_id="leaf_2",
            type=SegmentType.TEXT,
        ),
    ]

    with pytest.raises(ValueError):
        builder.build(leaves, n_levels=1)
