import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results

from rag_lib.loaders.structured import StructuredLoader
from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter
from rag_lib.core.domain import Document, Segment
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphRetriever
from rag_lib.llm.factory import get_llm

"""
E2E Example 05: DOCX Graph Workflow

Features Tested:
1. StructuredLoader: Loading DOCX into markdown-like structure.
2. RegexHierarchySplitter: Converting loaded Document -> hierarchical Segments.
3. EntityExtractor: Building a graph (entities + relations) from Segments.
4. GraphRetriever: Local graph-based retrieval over extracted nodes/edges.

Expected Results:
- Loading:
    - Input: docs/<docx file>
    - Output: List[Document]
- Segmentation:
    - Input: loaded markdown text
    - Output: List[Segment] with hierarchy metadata
- Extraction:
    - Output: Populated NetworkXGraphStore
- Retrieval:
    - Query: a concept from the DOCX (e.g. "zadacha")
    - Output: Entity context documents from graph traversal
"""


def _resolve_docx_path(docs_dir: Path) -> Optional[Path]:
    preferred_name = "Параметризованные задачи.docx"
    preferred_path = docs_dir / preferred_name
    if preferred_path.exists():
        return preferred_path

    candidates = sorted(docs_dir.glob("*.docx"))
    if candidates:
        return candidates[0]
    return None


def _build_graph_snapshot(graph_store: NetworkXGraphStore) -> Dict[str, Any]:
    nodes = []
    for node_id, attrs in graph_store.graph.nodes(data=True):
        node_record = {"id": node_id}
        node_record.update(attrs)
        nodes.append(node_record)

    edges = []
    for source, target, key, attrs in graph_store.graph.edges(keys=True, data=True):
        edge_record = {"source_id": source, "target_id": target, "edge_key": key}
        edge_record.update(attrs)
        edges.append(edge_record)

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def main():
    setup_environment()
    print_section("05. DOCX Graph Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    docx_path = _resolve_docx_path(docs_dir)
    if not docx_path:
        print(f"No DOCX files found in: {docs_dir}")
        return

    print_section("1. Loading DOCX")
    print(f"Loading {docx_path} using StructuredLoader...")
    loader = StructuredLoader(str(docx_path))
    docs = loader.load()

    if not docs:
        print("No content loaded from DOCX. Exiting.")
        return

    raw_text = docs[0].page_content
    print(f"Loaded {len(docs)} document(s).")
    print(f"Extracted {len(raw_text)} characters.")
    save_json_results(docs, "05_docx_graph", "loaded_documents")

    print_section("2. Structured Segmentation")
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

    print(f"Structured splitter created {len(segments)} segments.")
    if not segments:
        print("No segments produced after splitting. Exiting.")
        return

    print(f"Sample segment preview: {segments[0].content[:140]}...")
    print(f"Sample segment metadata: {segments[0].metadata}")
    save_json_results(segments, "05_docx_graph", "segments")

    print_section("3. Graph Extraction")
    graph_store = NetworkXGraphStore()
    llm = get_llm(provider="openai", model="gpt-4.1-nano", temperature=0, streaming=False)
    extractor = EntityExtractor(llm=llm, store=graph_store)

    max_graph_segments = min(10, len(segments))
    print(f"Extracting entities/relations from first {max_graph_segments} segments...")
    extractor.process_segments(segments[:max_graph_segments])

    num_nodes = graph_store.graph.number_of_nodes()
    num_edges = graph_store.graph.number_of_edges()
    print(f"Graph statistics: {num_nodes} nodes, {num_edges} edges.")

    graph_snapshot = _build_graph_snapshot(graph_store)
    save_json_results(graph_snapshot, "05_docx_graph", "graph_snapshot")

    if graph_snapshot["nodes"]:
        print("Sample extracted nodes:")
        for node in graph_snapshot["nodes"][:5]:
            print(f"- {node.get('label', node.get('id'))} ({node.get('type', 'unknown')})")
    else:
        print("No graph nodes extracted; retrieval will likely return no results.")

    print_section("4. Graph Retrieval")
    retriever = GraphRetriever(store=graph_store, mode="local", search_depth=1)
    query = "задача"
    print(f"Query: {query}")
    results = retriever.invoke(query)
    save_json_results(results, "05_docx_graph", "retrieved_results")

    if not results:
        print("No graph retrieval results.")
        return

    print(f"Top {min(10, len(results))} graph retrieval results:")
    for i, res in enumerate(results[:10], start=1):
        metadata = res.metadata or {}
        print(f"[{i}] {res.page_content[:180]}...")
        print(
            f"    source={metadata.get('source', 'graph')} "
            f"node_id={metadata.get('node_id', 'n/a')}"
        )


if __name__ == "__main__":
    main()
