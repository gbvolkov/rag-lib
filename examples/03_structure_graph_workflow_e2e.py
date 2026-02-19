import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from example_utils import _check_install, print_section, setup_environment

# Check for python-docx
try:
    import docx
except ImportError:
    _check_install("python-docx", "docx")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter
from rag_lib.core.indexer import Indexer
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.loaders.docx import DocXLoader
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.vectors.factory import create_vector_store


def main() -> None:
    setup_environment()

    # 1. Configuration
    docs_dir = Path(__file__).parent.parent / "docs"
    docx_file = docs_dir / "Параметризованные задачи.docx"
    collection_name = "graph_workflow_demo"

    print_section("1. Loading Structured Data (DOCX)")

    if not docx_file.exists():
        print(f"File {docx_file} not found!")
        return

    print(f"Loading {docx_file.name} using DocXLoader...")
    loader = DocXLoader(str(docx_file))
    docs = loader.load()
    print(f"Loaded {len(docs)} markdown document(s).")

    if not docs:
        return

    print(f"Sample markdown: {docs[0].page_content[:120]}...")
    print(f"Metadata: {docs[0].metadata}")

    splitter = RegexHierarchySplitter(
        patterns=[
            (1, r"^\s*#\s+(.+)$"),
            (2, r"^\s*##\s+(.+)$"),
            (3, r"^\s*###\s+(.+)$"),
            (4, r"^\s*####\s+(.+)$"),
        ],
        exclude_patterns=[r"^\s*$"],
        include_parent_content=False,
    )
    segments = splitter.split_documents(docs)
    print(f"Structured splitter produced {len(segments)} segment(s).")

    if not segments:
        return

    # 2. Graph Extraction
    print_section("2. Graph Extraction & Construction")
    print("Initializing NetworkXGraphStore (In-Memory)...")
    graph_store = NetworkXGraphStore()

    print("Initializing EntityExtractor with ChatOpenAI (gpt-3.5-turbo)...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    extractor = EntityExtractor(llm=llm, store=graph_store)

    max_graph_segments = 5
    print(f"Extracting entities from first {max_graph_segments} segments...")

    segments_to_process = segments[:max_graph_segments]
    extractor.process_segments(segments_to_process)

    num_nodes = graph_store.graph.number_of_nodes()
    num_edges = graph_store.graph.number_of_edges()
    print(f"\nGraph Statistics:\n  Nodes: {num_nodes}\n  Edges: {num_edges}")

    # 3. Indexing
    print_section("3. Indexing Segments")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    demo_db_path = "./chroma_demo_db_graph"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {collection_name}")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
    )

    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print("Indexing segments...")
    indexer.index(segments, batch_size=20)
    print("Indexing complete.")

    # 4. Retrieval (Multi-Modal: Vector + Graph)
    print_section("4. Retrieval (Vector + Graph)")

    query = "задача"
    print(f"Query: '{query}'")

    print("\n--- Vector Search Results ---")
    vector_results = vector_store.similarity_search(query, k=2)
    for i, res in enumerate(vector_results):
        print(f"[{i + 1}] {res.page_content[:150]}...")

    print("\n--- Graph Search Results ---")
    graph_nodes = graph_store.search_nodes(query)
    print(f"Found {len(graph_nodes)} nodes matching '{query}':")

    for node in graph_nodes[:3]:
        print(f"  Node: {node.label} ({node.type})")
        neighbors = graph_store.get_neighbors(node.id)
        if neighbors:
            print(f"    Neighbors ({len(neighbors)}): {[n.label for n in neighbors]}")
        else:
            print("    No neighbors.")


if __name__ == "__main__":
    main()