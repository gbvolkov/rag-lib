import pytest
from unittest.mock import MagicMock, call
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer

@pytest.fixture
def mock_vector_store():
    return MagicMock(spec=VectorStore)

@pytest.fixture
def mock_embeddings():
    return MagicMock(spec=Embeddings)

def test_indexer_embeds_content_when_no_summary(mock_vector_store, mock_embeddings):
    indexer = Indexer(mock_vector_store, mock_embeddings)
    
    seg = Segment(content="Original Content", type=SegmentType.TEXT)
    indexer.index([seg])
    
    # Assert add_texts called with content
    mock_vector_store.add_texts.assert_called_once()
    call_args = mock_vector_store.add_texts.call_args
    texts = call_args.kwargs['texts']
    metadatas = call_args.kwargs['metadatas']
    
    assert texts[0] == "Original Content"
    assert metadatas[0]["content"] == "Original Content"
    assert metadatas[0]["is_summary_embedding"] is False

def test_indexer_embeds_summary_when_present(mock_vector_store, mock_embeddings):
    indexer = Indexer(mock_vector_store, mock_embeddings)
    
    seg = Segment(
        content="Complex Table Content", 
        type=SegmentType.TABLE, 
        metadata={"summary": "Simple Summary"}
    )
    indexer.index([seg])
    
    # Assert add_texts called with SUMMARY
    mock_vector_store.add_texts.assert_called_once()
    call_args = mock_vector_store.add_texts.call_args
    texts = call_args.kwargs['texts']
    metadatas = call_args.kwargs['metadatas']
    
    assert texts[0] == "Simple Summary" # Embedded text is summary
    assert metadatas[0]["content"] == "Complex Table Content" # Payload is full content
    assert metadatas[0]["is_summary_embedding"] is True

def test_indexer_batching(mock_vector_store, mock_embeddings):
    indexer = Indexer(mock_vector_store, mock_embeddings)
    
    # Create 25 segments
    segments = [Segment(content=f"Seg {i}", type=SegmentType.TEXT) for i in range(25)]
    
    # Index with batch size 10
    indexer.index(segments, batch_size=10)
    
    # Expect 3 calls: 10, 10, 5
    assert mock_vector_store.add_texts.call_count == 3
    
    # Verify first batch
    first_call = mock_vector_store.add_texts.call_args_list[0]
    assert len(first_call.kwargs['texts']) == 10
    
    # Verify last batch
    last_call = mock_vector_store.add_texts.call_args_list[2]
    assert len(last_call.kwargs['texts']) == 5

def test_indexer_triggers_graph_extraction(mock_vector_store, mock_embeddings):
    # Mock EntityExtractor
    mock_extractor = MagicMock()
    
    indexer = Indexer(
        vector_store=mock_vector_store, 
        embeddings=mock_embeddings, 
        entity_extractor=mock_extractor
    )
    
    seg = Segment(content="Graph Content", type=SegmentType.TEXT)
    indexer.index([seg])
    
    # Assert process_segments called
    mock_extractor.process_segments.assert_called_once_with([seg])


def test_indexer_parent_segments_canonical(mock_vector_store, mock_embeddings):
    mock_doc_store = MagicMock()
    indexer = Indexer(mock_vector_store, mock_embeddings, doc_store=mock_doc_store)

    child = Segment(content="Child", type=SegmentType.TEXT)
    parent = Segment(content="Parent", type=SegmentType.TEXT)

    indexer.index([child], parent_segments=[parent])

    mock_doc_store.mset.assert_called_once()
    stored_pairs = mock_doc_store.mset.call_args.args[0]
    assert stored_pairs[0][0] == parent.segment_id
