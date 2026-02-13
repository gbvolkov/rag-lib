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
