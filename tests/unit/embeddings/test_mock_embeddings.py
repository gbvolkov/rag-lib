import pytest
import numpy as np
from rag_lib.embeddings.mock import MockEmbeddings

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def test_mock_embeddings_deterministic():
    model = MockEmbeddings(dimension=4)
    v1 = model.embed_query("apple")
    v2 = model.embed_query("apple")
    assert v1 == v2, "Embeddings for same text must be identical"
    
    v3 = model.embed_query("banana")
    assert v1 != v3, "Embeddings for different text should differ"

def test_mock_embeddings_semantic_properties():
    model = MockEmbeddings()
    
    vec_apple = model.embed_query("apple")
    vec_banana = model.embed_query("banana") # Should be close to apple
    vec_dog = model.embed_query("dog")       # Should be far from apple
    
    sim_fruit = cosine_similarity(vec_apple, vec_banana)
    sim_cross = cosine_similarity(vec_apple, vec_dog)
    
    print(f"\nSim(Apple, Banana): {sim_fruit}")
    print(f"Sim(Apple, Dog): {sim_cross}")
    
    assert sim_fruit > 0.8, "Related concepts should have high similarity"
    assert sim_cross < 0.5, "Unrelated concepts should have low similarity"

def test_embed_documents():
    model = MockEmbeddings()
    docs = ["apple", "dog", "space"]
    vectors = model.embed_documents(docs)
    assert len(vectors) == 3
    assert len(vectors[0]) == 4
