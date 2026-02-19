import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.loaders.docx import DocXLoader
from rag_lib.llm.factory import create_llm
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 05: DOCX Graph Workflow

Features Tested:
1. DocXLoader: Loading DOCX into markdown.
2. RegexHierarchySplitter: Converting loaded Document -> hierarchical Segments.
3. EntityExtractor: Building a graph (entities + relations) from Segments.
4. GraphRetriever: LightRAG-style graph retrieval (local/global/hybrid/mix).
"""


def _resolve_docx_path(docs_dir: Path) -> Optional[Path]:
    preferred_name = "????????????????? ??????.docx"
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


def main() -> None:
    setup_environment()
    print_section("05. DOCX Graph Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    docx_path = _resolve_docx_path(docs_dir)
    if not docx_path:
        print(f"No DOCX files found in: {docs_dir}")
        return

    print_section("1. Loading DOCX")
    print(f"Loading {docx_path} using DocXLoader...")
    loader = DocXLoader(str(docx_path))
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
    llm = create_llm(provider="openai", model_name="gpt-4.1-nano", temperature=0, streaming=False)
    extractor = EntityExtractor(llm=llm, store=graph_store)

    max_graph_segments = 10
    sample_start = 25
    sample_segments = segments[sample_start : sample_start + max_graph_segments]
    print(
        f"Extracting entities/relations from segments [{sample_start}:{sample_start + max_graph_segments}] "
        f"({len(sample_segments)} selected of {len(segments)})..."
    )
    extractor.process_segments(sample_segments)
    save_json_results(sample_segments, "05_docx_graph", "sample_segments")

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
    vector_texts = []
    vector_metadatas = []
    vector_ids = []
    for seg in sample_segments:
        if not seg.segment_id or not seg.content.strip():
            continue
        vector_texts.append(seg.content)
        vector_metadatas.append({"segment_id": seg.segment_id})
        vector_ids.append(seg.segment_id)

    if not vector_ids:
        print("No valid segments for vector indexing. Exiting.")
        return

    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    collection_name = f"docx_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
        cleanup=True,
    )
    vector_store.add_texts(texts=vector_texts, metadatas=vector_metadatas, ids=vector_ids)
    print(f"Indexed {len(vector_ids)} segments into Chroma collection '{collection_name}'.")

    modes = ["local", "mix", "global", "hybrid"]
    
    for mode in modes:
        print_section(f"5. Graph Retrieval (Mode: {mode})")
        if mode == "local":
            # Local: strict and precise around nearest entity neighborhood.
            graph_config = GraphQueryConfig(
                mode="local",
                max_hops=1,
                top_k_entities=6,
                top_k_relations=8,
                top_k_chunks=6,
                min_score=0.55,
                token_budget_entities=450,
                token_budget_relations=650,
                token_budget_chunks=2400,
                enable_keyword_extraction=True,
            )
        elif mode == "mix":
            # Mix: strongest chunk recall, but keep graph evidence filtered.
            graph_config = GraphQueryConfig(
                mode="mix",
                max_hops=1,
                top_k_entities=6,
                top_k_relations=10,
                top_k_chunks=8,
                min_score=0.50,
                token_budget_entities=450,
                token_budget_relations=700,
                token_budget_chunks=2350,
                enable_keyword_extraction=True,
            )
        elif mode == "global":
            # Global: relation/community-centric view with moderate filtering.
            graph_config = GraphQueryConfig(
                mode="global",
                max_hops=1,
                top_k_entities=8,
                top_k_relations=12,
                top_k_chunks=6,
                min_score=0.45,
                token_budget_entities=600,
                token_budget_relations=1200,
                token_budget_chunks=1700,
                enable_keyword_extraction=True,
            )
        else:
            # Hybrid: balanced graph coverage + chunk evidence.
            graph_config = GraphQueryConfig(
                mode="hybrid",
                max_hops=1,
                top_k_entities=8,
                top_k_relations=10,
                top_k_chunks=7,
                min_score=0.50,
                token_budget_entities=550,
                token_budget_relations=900,
                token_budget_chunks=2000,
                enable_keyword_extraction=True,
            )

        retriever = GraphRetriever(
            vector_store=vector_store,
            graph_store=graph_store,
            config=graph_config,
            llm=llm,
        )

        queries = ["?????? ???????????", "???????????"]
        for query in queries:
            print(f"Query: {query}")
            results = retriever.invoke(query)
            save_json_results(results, "05_docx_graph", f"retrieved_results_mode_{mode}_q_{query}")

            if not results:
                print(f"No graph retrieval results for {query}.")
                continue

            print(f"Top {min(10, len(results))} graph retrieval results for {query}:")
            for i, res in enumerate(results[:10], start=1):
                metadata = res.metadata or {}
                print(f"[{i}] {res.page_content[:180]}...")
                print(
                    f"    kind={metadata.get('retrieval_kind', 'n/a')} "
                    f"score={metadata.get('score', 0):.3f} "
                    f"entity_id={metadata.get('entity_id', 'n/a')} "
                    f"edge_id={metadata.get('edge_id', 'n/a')} "
                    f"source_segment_id={metadata.get('source_segment_id', 'n/a')}"
                )


if __name__ == "__main__":
    main()
