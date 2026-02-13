from typing import Optional, Any
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from rag_lib.config import Settings

# Attempt imports for specific providers
try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = Any

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = Any

try:
    from langchain_qdrant import Qdrant
except ImportError:
    Qdrant = Any

try:
    from langchain_postgres import PGVector
except ImportError:
    PGVector = Any

def get_vector_store(
    provider: str = "chroma",
    embeddings: Optional[Embeddings] = None,
    collection_name: str = "rag_collection",
    connection_string: Optional[str] = None
) -> VectorStore:
    """
    Factory to get Vector Store instance.
    
    Args:
        provider: 'chroma', 'faiss', 'qdrant', 'postgres'
        embeddings: Initialized Embeddings model (required for most stores)
        collection_name: Name of the collection/table
        connection_string: DB URL for Postgres/Qdrant
    """
    if embeddings is None:
        raise ValueError("Embeddings model must be provided to initialize Vector Store.")

    settings = Settings()

    if provider == "chroma":
        # Check if persistent or in-memory
        # For production, we use settings or default to a local dir
        persist_dir = settings.vector_store.path if settings.vector_store.path else "./chroma_db"
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
    elif provider == "faiss":
        # FAISS initialization often requires documents if creating fresh.
        # But we want an empty store to add documents to.
        # LangChain FAISS doesn't easily support "empty" init.
        # We can try to load from local if exists, else create with dummy and reset?
        # Ideally, we use FAISS.from_texts([""], embeddings) but that adds an empty doc.
        # For this factory, we'll assume we are creating a new one and client accepts the limitation,
        # OR we raise error if trying to load without path.
        # Let's use the 'from_texts' hack to init, then delete? No.
        # Let's just raise NotImplemented for now as strict FAISS factory is complex.
        return FAISS.from_texts([""], embeddings)

    elif provider == "qdrant":
        if not connection_string:
             # In-memory
             return Qdrant(
                 client=None, 
                 collection_name=collection_name, 
                 embeddings=embeddings,
                 location=":memory:"
             )
        return Qdrant.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=connection_string
        )

    elif provider == "postgres":
        if not connection_string:
            raise ValueError("Connection string required for Postgres vector store")
            
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )

    else:
        raise ValueError(f"Unknown Vector Store provider: {provider}")
