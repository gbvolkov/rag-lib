import pytest
import os
import shutil
import json
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.store import JsonFileStore, LocalPickleStore

# Setup temporary directory for tests
TEST_STORE_DIR = "tests/data_registry/store_test"

@pytest.fixture
def clean_store_dir():
    if os.path.exists(TEST_STORE_DIR):
        shutil.rmtree(TEST_STORE_DIR)
    os.makedirs(TEST_STORE_DIR)
    yield
    if os.path.exists(TEST_STORE_DIR):
        shutil.rmtree(TEST_STORE_DIR)

# --- JsonFileStore Tests ---

def test_json_store_basic_ops(clean_store_dir):
    """
    Test basic KV operations: mset, mget, mdelete, yield_keys
    """
    store_path = os.path.join(TEST_STORE_DIR, "segments.json")
    store = JsonFileStore(file_path=store_path)
    
    seg1 = Segment(content="Content 1", segment_id="1", type=SegmentType.TEXT)
    seg2 = Segment(content="Content 2", segment_id="2", type=SegmentType.TEXT)
    
    # 1. mset (Multi-Set)
    store.mset([("1", seg1), ("2", seg2)])
    
    # Verify file existence
    assert os.path.exists(store_path)
    
    # 2. mget (Multi-Get)
    retrieved = store.mget(["1", "2", "3"])
    assert len(retrieved) == 3
    assert retrieved[0].content == "Content 1"
    assert retrieved[1].content == "Content 2"
    assert retrieved[2] is None # Key "3" doesn't exist
    
    # 3. yield_keys
    keys = list(store.yield_keys())
    assert "1" in keys
    assert "2" in keys
    assert len(keys) == 2

    # 4. mdelete
    store.mdelete(["1"])
    retrieved_after_del = store.mget(["1"])
    assert retrieved_after_del[0] is None

def test_json_store_persistence(clean_store_dir):
    """
    Test that data persists after store object is recreated.
    """
    store_path = os.path.join(TEST_STORE_DIR, "persist.json")
    store1 = JsonFileStore(file_path=store_path)
    seg = Segment(content="Persistent", segment_id="p1")
    store1.mset([("p1", seg)])
    
    # Reload
    store2 = JsonFileStore(file_path=store_path)
    result = store2.mget(["p1"])
    assert result[0].content == "Persistent"


# --- LocalFileStore (Pickle/Bytes) Tests ---
# Note: LangChain's LocalFileStore usually stores bytes. 
# We might need a wrapper "SegmentStore" that uses LocalFileStore backend but manual serialization?
# Or we implement a simple PickleFileStore that behaves like BaseStore[str, Segment].
# Let's test "PickleFileStore" concept.

def test_pickle_store_ops(clean_store_dir):
    """
    Test PickleFileStore for binary serialization.
    """
    import pickle
    store_path = os.path.join(TEST_STORE_DIR, "segments.pkl")
    # For this test, we assume we implement a simple LocalPickleStore
    from rag_lib.core.store import LocalPickleStore 
    
    store = LocalPickleStore(file_path=store_path)
    seg = Segment(content="Binary Content", segment_id="b1")
    
    store.mset([("b1", seg)])
    
    # Reload
    store2 = LocalPickleStore(file_path=store_path)
    result = store2.mget(["b1"])
    assert result[0].content == "Binary Content"
    assert result[0].type == SegmentType.TEXT
