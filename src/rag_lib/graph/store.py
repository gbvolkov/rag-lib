from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Optional

import networkx as nx
from langchain_core.embeddings import Embeddings

from rag_lib.core.logger import logger
from rag_lib.graph.domain import GraphEdge, GraphNode


@dataclass
class GraphExpansionResult:
    """
    Expanded graph neighborhood rooted at one or more seed nodes.
    """

    nodes: List[GraphNode]
    edges: List[GraphEdge]
    hop_by_node: Dict[str, int] = field(default_factory=dict)


class BaseGraphStore(ABC):
    """
    Abstract interface for Graph storage.
    Allows for different backends (NetworkX, Neo4j, FalkorDB, etc.)
    """

    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        pass

    @abstractmethod
    def search_nodes(self, query: str) -> List[GraphNode]:
        """Simple keyword search on node labels/descriptions."""
        pass

    def search_nodes_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphNode, float]]:
        """
        Default fallback implementation when backend-specific ranking is unavailable.
        """
        nodes = self.search_nodes(query)
        return [(node, 1.0) for node in nodes[: max(0, top_k)]]

    def search_edges_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphEdge, float]]:
        """
        Default fallback implementation when backend-specific edge search is unavailable.
        """
        return []

    def expand_subgraph(
        self, seed_node_ids: List[str], max_hops: int, max_nodes: Optional[int] = None
    ) -> GraphExpansionResult:
        """
        Default fallback expansion when backend-specific implementation is unavailable.
        """
        if max_hops < 0:
            return GraphExpansionResult(nodes=[], edges=[], hop_by_node={})

        visited: Dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()
        for node_id in seed_node_ids:
            node = self.get_node(node_id)
            if node is None:
                continue
            if node_id in visited:
                continue
            visited[node_id] = 0
            queue.append((node_id, 0))

        while queue:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor in self.get_neighbors(current, depth=1):
                if neighbor.id in visited:
                    continue
                if max_nodes is not None and len(visited) >= max_nodes:
                    break
                visited[neighbor.id] = hop + 1
                queue.append((neighbor.id, hop + 1))

        nodes: List[GraphNode] = []
        for node_id in sorted(visited, key=lambda nid: (visited[nid], nid)):
            node = self.get_node(node_id)
            if node is not None:
                nodes.append(node)
        return GraphExpansionResult(nodes=nodes, edges=[], hop_by_node=visited)

    def get_community_summaries(
        self, top_k: int, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Optional graph-native community summaries.
        """
        return []

    def get_source_segment_ids_for_entities(self, entity_ids: List[str]) -> List[str]:
        source_ids: List[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            node = self.get_node(entity_id)
            if node is None or not node.source_segment_id:
                continue
            if node.source_segment_id in seen:
                continue
            seen.add(node.source_segment_id)
            source_ids.append(node.source_segment_id)
        return source_ids

    def get_source_segment_ids_for_edges(self, edge_ids: List[str]) -> List[str]:
        return []

    def get_node_prior(self, node_id: str) -> float:
        """
        Optional lightweight centrality prior, normalized to [0, 1].
        """
        return 0.0

    def get_edge_prior(self, edge_id: str) -> float:
        """
        Optional lightweight edge prior, normalized to [0, 1].
        """
        return 0.0

    # Async Interface
    async def aadd_node(self, node: GraphNode) -> None:
        self.add_node(node)

    async def aadd_edge(self, edge: GraphEdge) -> None:
        self.add_edge(edge)

    async def aget_node(self, node_id: str) -> Optional[GraphNode]:
        return self.get_node(node_id)

    async def aget_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        return self.get_neighbors(node_id, depth)

    async def asearch_nodes(self, query: str) -> List[GraphNode]:
        return self.search_nodes(query)

    async def asearch_nodes_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphNode, float]]:
        return self.search_nodes_hybrid(query=query, top_k=top_k, embedder=embedder)

    async def asearch_edges_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphEdge, float]]:
        return self.search_edges_hybrid(query=query, top_k=top_k, embedder=embedder)

    async def aexpand_subgraph(
        self, seed_node_ids: List[str], max_hops: int, max_nodes: Optional[int] = None
    ) -> GraphExpansionResult:
        return self.expand_subgraph(seed_node_ids, max_hops, max_nodes)

    async def aget_community_summaries(
        self, top_k: int, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.get_community_summaries(top_k=top_k, query=query)

    async def aget_source_segment_ids_for_entities(self, entity_ids: List[str]) -> List[str]:
        return self.get_source_segment_ids_for_entities(entity_ids)

    async def aget_source_segment_ids_for_edges(self, edge_ids: List[str]) -> List[str]:
        return self.get_source_segment_ids_for_edges(edge_ids)

    async def aget_node_prior(self, node_id: str) -> float:
        return self.get_node_prior(node_id)

    async def aget_edge_prior(self, edge_id: str) -> float:
        return self.get_edge_prior(edge_id)


class NetworkXGraphStore(BaseGraphStore):
    """
    In-memory graph store using NetworkX.
    Good for small-to-medium graphs or prototyping.
    """

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._edge_id_to_key: Dict[str, tuple[str, str, int]] = {}
        logger.info("Initialized NetworkXGraphStore")

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    def add_node(self, node: GraphNode) -> None:
        self._graph.add_node(
            node.id,
            type=node.type,
            label=node.label,
            description=node.description,
            properties=node.properties,
            source_segment_id=node.source_segment_id,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        properties = dict(edge.properties or {})
        key = self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            properties=properties,
            source_segment_id=edge.source_segment_id,
        )
        edge_id = str(
            properties.get(
                "edge_id",
                f"{edge.source_id}|{edge.relation_type}|{edge.target_id}|{key}",
            )
        )
        self._graph[edge.source_id][edge.target_id][key]["edge_id"] = edge_id
        self._edge_id_to_key[edge_id] = (edge.source_id, edge.target_id, key)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        if not self._graph.has_node(node_id):
            return None

        data = self._graph.nodes[node_id]
        return GraphNode(
            id=node_id,
            type=data.get("type", "unknown"),
            label=data.get("label", node_id),
            description=data.get("description"),
            properties=dict(data.get("properties", {})),
            source_segment_id=data.get("source_segment_id"),
        )

    def _adjacent_node_ids(self, node_id: str) -> List[str]:
        if not self._graph.has_node(node_id):
            return []
        adjacent = set(self._graph.successors(node_id))
        adjacent.update(self._graph.predecessors(node_id))
        return sorted(adjacent)

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        if depth < 1 or not self._graph.has_node(node_id):
            return []

        visited = {node_id}
        frontier = {node_id}
        result_ids: set[str] = set()

        for _ in range(depth):
            next_frontier: set[str] = set()
            for current in frontier:
                for neighbor_id in self._adjacent_node_ids(current):
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    result_ids.add(neighbor_id)
                    next_frontier.add(neighbor_id)
            if not next_frontier:
                break
            frontier = next_frontier

        result: List[GraphNode] = []
        for neighbor_id in sorted(result_ids):
            node = self.get_node(neighbor_id)
            if node is not None:
                result.append(node)
        return result

    def _tokenize(self, text: str) -> List[str]:
        import re

        return re.findall(r"[A-Za-z0-9\u0400-\u04FF]+", text.lower())

    def _lexical_score(self, query: str, text: str) -> float:
        query_tokens = set(self._tokenize(query))
        text_tokens = set(self._tokenize(text))
        if not query_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(text_tokens)) / max(len(query_tokens), 1)
        exact = 1.0 if query.strip().lower() in text.lower() and query.strip() else 0.0
        stem_hits = 0
        for q_token in query_tokens:
            for t_token in text_tokens:
                if len(q_token) < 4 or len(t_token) < 4:
                    continue
                if q_token in t_token or t_token in q_token or q_token[:4] == t_token[:4]:
                    stem_hits += 1
                    break
        stem = stem_hits / max(len(query_tokens), 1)
        return min(1.0, 0.55 * overlap + 0.25 * exact + 0.20 * stem)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_nodes(self, query: str) -> List[GraphNode]:
        results: List[GraphNode] = []
        q = query.lower()
        for node_id, data in self._graph.nodes(data=True):
            label = str(data.get("label", "")).lower()
            desc = str(data.get("description", "") or "").lower()
            if not q or q in label or q in desc:
                node = self.get_node(node_id)
                if node is not None:
                    results.append(node)
        return results

    def _node_degree_norm(self, node_id: str, max_degree: int) -> float:
        if max_degree <= 0:
            return 0.0
        degree = self._graph.degree(node_id)
        return float(degree) / float(max_degree)

    def search_nodes_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphNode, float]]:
        if top_k <= 0:
            return []

        node_ids = list(self._graph.nodes())
        if not node_ids:
            return []

        max_degree = max((self._graph.degree(nid) for nid in node_ids), default=0)
        query_embedding: Optional[List[float]] = None
        node_embeddings: Optional[List[List[float]]] = None

        if embedder is not None and query.strip():
            try:
                query_embedding = embedder.embed_query(query)
                node_texts = [
                    f"{self._graph.nodes[nid].get('label', nid)} {self._graph.nodes[nid].get('description', '') or ''}"
                    for nid in node_ids
                ]
                node_embeddings = embedder.embed_documents(node_texts)
            except Exception as exc:
                logger.debug(f"Node hybrid embedding scoring skipped: {exc}")
                query_embedding = None
                node_embeddings = None

        scored: List[tuple[GraphNode, float]] = []
        for idx, node_id in enumerate(node_ids):
            node = self.get_node(node_id)
            if node is None:
                continue
            text = f"{node.label} {node.description or ''}".strip()
            lexical = self._lexical_score(query, text) if query.strip() else 0.0
            prior = self._node_degree_norm(node_id, max_degree)
            embed_score = 0.0
            if query_embedding is not None and node_embeddings is not None and idx < len(node_embeddings):
                embed_score = max(0.0, self._cosine_similarity(query_embedding, node_embeddings[idx]))
            if query.strip() and lexical <= 0.0 and embed_score <= 0.0:
                continue
            score = 0.65 * lexical + 0.20 * embed_score + 0.15 * prior
            if not query.strip():
                score = prior
            if score <= 0.0:
                continue
            scored.append((node, min(1.0, score)))

        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:top_k]

    def _edge_to_model(self, source: str, target: str, key: int, data: Dict[str, Any]) -> GraphEdge:
        relation_type = str(data.get("relation_type", "RELATED_TO"))
        edge_id = str(data.get("edge_id", f"{source}|{relation_type}|{target}|{key}"))
        properties = dict(data.get("properties", {}))
        properties["edge_id"] = edge_id
        properties.setdefault("source_label", self._graph.nodes[source].get("label", source))
        properties.setdefault("target_label", self._graph.nodes[target].get("label", target))
        return GraphEdge(
            source_id=source,
            target_id=target,
            relation_type=relation_type,
            weight=float(data.get("weight", 1.0)),
            properties=properties,
            source_segment_id=data.get("source_segment_id"),
        )

    def search_edges_hybrid(
        self, query: str, top_k: int, embedder: Optional[Embeddings] = None
    ) -> List[tuple[GraphEdge, float]]:
        if top_k <= 0:
            return []

        edge_rows: List[tuple[str, str, int, Dict[str, Any]]] = [
            (source, target, key, data)
            for source, target, key, data in self._graph.edges(keys=True, data=True)
        ]
        if not edge_rows:
            return []

        abs_weights = [abs(float(data.get("weight", 1.0))) for _, _, _, data in edge_rows]
        max_abs_weight = max(abs_weights, default=1.0)
        query_embedding: Optional[List[float]] = None
        edge_embeddings: Optional[List[List[float]]] = None

        if embedder is not None and query.strip():
            try:
                query_embedding = embedder.embed_query(query)
                edge_texts = []
                for source, target, _, data in edge_rows:
                    source_label = self._graph.nodes[source].get("label", source)
                    target_label = self._graph.nodes[target].get("label", target)
                    relation_type = data.get("relation_type", "RELATED_TO")
                    edge_texts.append(f"{source_label} {relation_type} {target_label}")
                edge_embeddings = embedder.embed_documents(edge_texts)
            except Exception as exc:
                logger.debug(f"Edge hybrid embedding scoring skipped: {exc}")
                query_embedding = None
                edge_embeddings = None

        scored: List[tuple[GraphEdge, float]] = []
        for idx, (source, target, key, data) in enumerate(edge_rows):
            edge = self._edge_to_model(source, target, key, data)
            source_label = self._graph.nodes[source].get("label", source)
            target_label = self._graph.nodes[target].get("label", target)
            text = f"{source_label} {edge.relation_type} {target_label}"
            lexical = self._lexical_score(query, text) if query.strip() else 0.0
            weight_score = abs(edge.weight) / max_abs_weight if max_abs_weight > 0 else 0.0
            endpoint_prior = (self.get_node_prior(source) + self.get_node_prior(target)) / 2.0
            embed_score = 0.0
            if query_embedding is not None and edge_embeddings is not None and idx < len(edge_embeddings):
                embed_score = max(0.0, self._cosine_similarity(query_embedding, edge_embeddings[idx]))
            if query.strip() and lexical <= 0.0 and embed_score <= 0.0:
                continue
            score = 0.55 * lexical + 0.20 * embed_score + 0.15 * weight_score + 0.10 * endpoint_prior
            if not query.strip():
                score = 0.6 * weight_score + 0.4 * endpoint_prior
            if score <= 0.0:
                continue
            scored.append((edge, min(1.0, score)))

        scored.sort(
            key=lambda item: (
                -item[1],
                str(item[0].properties.get("edge_id", "")),
                item[0].source_id,
                item[0].target_id,
            )
        )
        return scored[:top_k]

    def expand_subgraph(
        self, seed_node_ids: List[str], max_hops: int, max_nodes: Optional[int] = None
    ) -> GraphExpansionResult:
        if max_hops < 0:
            return GraphExpansionResult(nodes=[], edges=[], hop_by_node={})

        hop_by_node: Dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for node_id in seed_node_ids:
            if not self._graph.has_node(node_id) or node_id in hop_by_node:
                continue
            hop_by_node[node_id] = 0
            queue.append((node_id, 0))

        edge_by_id: Dict[str, GraphEdge] = {}

        while queue:
            current, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor in self._adjacent_node_ids(current):
                next_hop = hop + 1
                if neighbor not in hop_by_node:
                    if max_nodes is not None and len(hop_by_node) >= max_nodes:
                        continue
                    hop_by_node[neighbor] = next_hop
                    queue.append((neighbor, next_hop))
                elif next_hop < hop_by_node[neighbor]:
                    hop_by_node[neighbor] = next_hop

                forward = self._graph.get_edge_data(current, neighbor, default={})
                reverse = self._graph.get_edge_data(neighbor, current, default={})
                for source, target, edge_map in (
                    (current, neighbor, forward),
                    (neighbor, current, reverse),
                ):
                    for key, data in edge_map.items():
                        edge = self._edge_to_model(source, target, key, data)
                        edge_id = str(
                            edge.properties.get(
                                "edge_id",
                                f"{edge.source_id}|{edge.relation_type}|{edge.target_id}|{key}",
                            )
                        )
                        edge_by_id[edge_id] = edge

        ordered_node_ids = sorted(hop_by_node, key=lambda nid: (hop_by_node[nid], nid))
        nodes: List[GraphNode] = []
        for node_id in ordered_node_ids:
            node = self.get_node(node_id)
            if node is not None:
                nodes.append(node)

        edges = sorted(
            edge_by_id.values(),
            key=lambda edge: str(
                edge.properties.get(
                    "edge_id", f"{edge.source_id}|{edge.relation_type}|{edge.target_id}"
                )
            ),
        )
        return GraphExpansionResult(nodes=nodes, edges=edges, hop_by_node=hop_by_node)

    def get_community_summaries(
        self, top_k: int, query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        q = (query or "").strip().lower()
        summaries: List[Dict[str, Any]] = []
        for node_id, data in self._graph.nodes(data=True):
            is_summary = bool(data.get("is_community_summary")) or str(data.get("type", "")).lower() in {
                "community_summary",
                "community",
            }
            if not is_summary:
                continue
            summary = str(data.get("description", "") or "")
            if q and q not in summary.lower() and q not in str(data.get("label", "")).lower():
                continue
            summaries.append(
                {
                    "community_id": data.get("community_id", node_id),
                    "summary": summary,
                    "score": 1.0,
                    "source_segment_id": data.get("source_segment_id"),
                }
            )

        summaries.sort(key=lambda item: str(item.get("community_id", "")))
        return summaries[:top_k]

    def get_source_segment_ids_for_entities(self, entity_ids: List[str]) -> List[str]:
        source_ids: List[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            node = self.get_node(entity_id)
            if node is None or not node.source_segment_id:
                continue
            if node.source_segment_id in seen:
                continue
            seen.add(node.source_segment_id)
            source_ids.append(node.source_segment_id)
        return source_ids

    def get_source_segment_ids_for_edges(self, edge_ids: List[str]) -> List[str]:
        source_ids: List[str] = []
        seen: set[str] = set()
        for edge_id in edge_ids:
            key_tuple = self._edge_id_to_key.get(edge_id)
            if key_tuple is None:
                continue
            source, target, key = key_tuple
            edge_data = self._graph.get_edge_data(source, target, key, default={})
            source_segment_id = edge_data.get("source_segment_id")
            if not source_segment_id or source_segment_id in seen:
                continue
            seen.add(source_segment_id)
            source_ids.append(source_segment_id)
        return source_ids

    def get_node_prior(self, node_id: str) -> float:
        if not self._graph.has_node(node_id):
            return 0.0
        max_degree = max((self._graph.degree(nid) for nid in self._graph.nodes()), default=0)
        return self._node_degree_norm(node_id, max_degree)

    def get_edge_prior(self, edge_id: str) -> float:
        key_tuple = self._edge_id_to_key.get(edge_id)
        if key_tuple is None:
            return 0.0
        source, target, key = key_tuple
        edge_data = self._graph.get_edge_data(source, target, key, default={})
        if not edge_data:
            return 0.0
        weight = abs(float(edge_data.get("weight", 1.0)))
        all_weights = [
            abs(float(data.get("weight", 1.0)))
            for _, _, _, data in self._graph.edges(keys=True, data=True)
        ]
        max_weight = max(all_weights, default=1.0)
        if max_weight <= 0:
            return 0.0
        return min(1.0, weight / max_weight)

    def save_to_file(self, path: str):
        nx.write_gml(self._graph, path)

    def load_from_file(self, path: str):
        self._graph = nx.read_gml(path)
        self._edge_id_to_key = {}
        for source, target, key, data in self._graph.edges(keys=True, data=True):
            edge_id = str(
                data.get(
                    "edge_id",
                    f"{source}|{data.get('relation_type', 'RELATED_TO')}|{target}|{key}",
                )
            )
            self._graph[source][target][key]["edge_id"] = edge_id
            self._edge_id_to_key[edge_id] = (source, target, key)


def create_graph_store(
    provider: Optional[str] = None,
    uri: Optional[str] = None,
    auth: Optional[tuple[str, str]] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
) -> BaseGraphStore:
    """
    Factory to create a graph store backend.

    Args:
        provider: Backend provider ("networkx" or "neo4j"). Falls back to settings.
        uri: Neo4j URI (e.g. "bolt://localhost:7687").
        auth: Neo4j auth tuple of (username, password). If provided, overrides username/password.
        username: Neo4j username when auth is not provided.
        password: Neo4j password when auth is not provided.
        database: Neo4j database name. Defaults to "neo4j" via settings.
    """
    from rag_lib.config import Settings

    settings = Settings()
    resolved_provider = (provider or settings.graph_store.provider or "networkx").strip().lower()

    if resolved_provider == "networkx":
        return NetworkXGraphStore()

    if resolved_provider == "neo4j":
        resolved_uri = uri or settings.graph_store.uri
        resolved_database = database or settings.graph_store.database or "neo4j"

        resolved_auth: Optional[tuple[str, str]] = None
        if auth is not None:
            if len(auth) != 2 or not auth[0] or not auth[1]:
                raise ValueError("`auth` must be a tuple of (username, password).")
            resolved_auth = auth
        else:
            resolved_username = username if username is not None else settings.graph_store.username
            resolved_password = password if password is not None else settings.graph_store.password
            if resolved_username and resolved_password:
                resolved_auth = (resolved_username, resolved_password)

        if not resolved_uri:
            raise ValueError("Neo4j provider requires `uri` (or GRAPH_URI).")
        if resolved_auth is None:
            raise ValueError(
                "Neo4j provider requires `auth` or `username`/`password` "
                "(or GRAPH_USERNAME/GRAPH_PASSWORD)."
            )

        return Neo4jGraphStore(uri=resolved_uri, auth=resolved_auth, database=resolved_database)

    raise ValueError(f"Unknown Graph Store provider: {resolved_provider}")


from rag_lib.graph.neo4j_store import Neo4jGraphStore


__all__ = [
    "BaseGraphStore",
    "GraphExpansionResult",
    "NetworkXGraphStore",
    "Neo4jGraphStore",
    "create_graph_store",
]
