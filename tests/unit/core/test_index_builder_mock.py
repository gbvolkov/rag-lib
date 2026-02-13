import pytest
from unittest.mock import MagicMock, AsyncMock, call
from rag_lib.core.domain import Segment, SegmentType
# We will create this class in the next step
from rag_lib.core.index_builder import IndexBuilder

def test_builder_dual_storage_flow():
    """
    Verify that segments go to DocStore AND VectorStore (with reduced payload).
    """
    # Mocks
    mock_vec_store = MagicMock()
    mock_doc_store = MagicMock()
    
    # Setup Builder
    builder = IndexBuilder(vector_store=mock_vec_store, doc_store=mock_doc_store)
    
    # Input Data
    seg1 = Segment(
        content="Detailed Content 1", 
        segment_id="s1", 
        type=SegmentType.TEXT,
        metadata={"title": "Doc 1", "summary": "Sum 1"}
    )
    seg2 = Segment(
        content="Detailed Content 2", 
        segment_id="s2", 
        type=SegmentType.TEXT,
        metadata={"title": "Doc 2"} # No summary
    )
    
    # Execute Build
    builder.build([seg1, seg2])
    
    # Verification 1: Doc Store receives FULL segments
    # mset should be called with [("s1", seg1), ("s2", seg2)]
    assert mock_doc_store.mset.called
    args = mock_doc_store.mset.call_args[0][0] # First arg is list of tuples
    assert len(args) == 2
    assert args[0][0] == "s1"
    assert args[0][1] == seg1
    
    # Verification 2: Vector Store receives CHUNKS/Payloads
    # logic similar to Indexer:
    # Seg1 has summary -> Embed "Sum 1", store small payload
    # Seg2 no summary -> Embed "Detailed Content 2", store small payload
    
    assert mock_vec_store.add_texts.called
    call_kwargs = mock_vec_store.add_texts.call_args[1]
    texts = call_kwargs['texts']
    metadatas = call_kwargs['metadatas']
    ids = call_kwargs['ids']
    
    # IDs match
    assert ids == ["s1", "s2"]
    
    # Texts to embed
    assert texts[0] == "Sum 1"
    assert texts[1] == "Detailed Content 2"
    
    # Payloads (Lightweight)
    # Should contain 'segment_id' to link back
    # Should NOT contain full content if we want strictly lightweight? 
    # User request was "store index AND related segments (separately from chunks)."
    # "Full chunks metadata to be stored, but embeddings will be applied to content only."
    # The previous implementation stored full content in vector payload to allow retrieval without lookups.
    # But here we have dual storage. 
    # If we want retrieval to be efficient, we might still strip content from vector payload if it's huge?
    # BUT user comment: "Full chunks metadata to be stored".
    # This implies we keep metadata but maybe not the raw 'content' field if it's redundant?
    # Let's assume for now we keep 'content' in vector payload ONLY if it's the embedding target.
    # If we embed summary, maybe we don't need full content in vector DB?
    # The 'Retriever' later will use 'segment_id' to fetch full doc from DocStore.
    
    # Let's assert that we key behavior: segment_id IS present.
    assert metadatas[0]["segment_id"] == "s1"
    assert metadatas[1]["segment_id"] == "s2"
    
    # And check origin
    assert metadatas[0]["title"] == "Doc 1"

@pytest.mark.asyncio
async def test_abuilder_flow():
    mock_vec_store = MagicMock()
    mock_vec_store.aadd_texts = AsyncMock()
    mock_doc_store = MagicMock()
    mock_doc_store.amset = AsyncMock() # BaseStore usually has mset, need to check if async available or we wrap it
    # LangChain BaseStore doesn't strictly mandate async mset? 
    # Actually it does `amset`.
    
    # Mock DocumentStore needs to support amset for this test if IndexBuilder uses it.
    
    builder = IndexBuilder(vector_store=mock_vec_store, doc_store=mock_doc_store)
    
    seg = Segment(content="Async", segment_id="a1")
    await builder.abuild([seg])
    
    assert mock_doc_store.amset.called
    assert mock_vec_store.aadd_texts.called
