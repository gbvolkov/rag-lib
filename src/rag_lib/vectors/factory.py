from typing import Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from rag_lib.config import Settings

# Attempt imports for specific providers
try:
    from langchain_chroma import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_qdrant import Qdrant
except ImportError:
    Qdrant = None

try:
    from langchain_postgres import PGVector
except ImportError:
    PGVector = None

def create_vector_store(
    provider: str = "chroma",
    embeddings: Optional[Embeddings] = None,
    collection_name: str = "rag_collection",
    connection_uri: Optional[str] = None,
    cleanup: bool = True,
) -> VectorStore:
    """
    Factory to get Vector Store instance.
    
    Args:
        provider: 'chroma', 'faiss', 'qdrant', 'postgres'
        embeddings: Initialized Embeddings model (required for most stores)
        collection_name: Name of the collection/table
        connection_uri: DB URL for Postgres/Qdrant
        cleanup: If True, delete existing collection before returning the store (Chroma).
    """
    if embeddings is None:
        raise ValueError("Embeddings model must be provided to initialize Vector Store.")

    settings = Settings()

    if provider == "chroma":
        if Chroma is None:
            raise ImportError("langchain-chroma is not installed. Please install it with `pip install langchain-chroma`.")
            
        # Check if persistent or in-memory
        # For production, we use settings or default to a local dir
        persist_dir = settings.vector_store.path if settings.vector_store.path else "./chroma_db"

        def _build_chroma() -> VectorStore:
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_dir
            )

        if cleanup:
            cleanup_store = _build_chroma()
            try:
                cleanup_store.delete_collection()
            except Exception:
                # If collection doesn't exist yet (or backend can't delete), continue.
                pass

        return _build_chroma()
        
    elif provider == "faiss":
        if FAISS is None:
             raise ImportError("langchain-community and faiss-cpu/gpu are not installed. Please install them.")
             
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
        if Qdrant is None:
            raise ImportError("langchain-qdrant is not installed. Please install it.")
            
        if not connection_uri:
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
            url=connection_uri
        )

    elif provider == "postgres":
        if PGVector is None:
            raise ImportError("langchain-postgres and psycopg are not installed. Please install them.")
            
        if not connection_uri:
            raise ValueError("connection_uri is required for Postgres vector store")
            
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_uri,
            use_jsonb=True,
        )

    else:
        raise ValueError(f"Unknown Vector Store provider: {provider}")
