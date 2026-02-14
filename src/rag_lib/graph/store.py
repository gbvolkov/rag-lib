from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import networkx as nx
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.core.logger import logger

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
        """Simple keyword search on node labels/descriptions"""
        pass
        
    # Async Interface
    async def aadd_node(self, node: GraphNode) -> None:
        """Async version of add_node"""
        self.add_node(node)

    async def aadd_edge(self, edge: GraphEdge) -> None:
        """Async version of add_edge"""
        self.add_edge(edge)

    async def aget_node(self, node_id: str) -> Optional[GraphNode]:
        """Async version of get_node"""
        return self.get_node(node_id)
        
    async def aget_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        """Async version of get_neighbors"""
        return self.get_neighbors(node_id, depth)
        
    async def asearch_nodes(self, query: str) -> List[GraphNode]:
        """Async version of search_nodes"""
        return self.search_nodes(query)


class NetworkXGraphStore(BaseGraphStore):
    """
    In-memory graph store using NetworkX.
    Good for small-to-medium graphs or prototyping.
    """
    def __init__(self):
        self._graph = nx.MultiDiGraph()
        logger.info("Initialized NetworkXGraphStore")

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    def add_node(self, node: GraphNode) -> None:
        self._graph.add_node(
            node.id, 
            type=node.type, 
            label=node.label, 
            description=node.description,
            properties=node.properties,
            source_segment_id=node.source_segment_id
        )

    def add_edge(self, edge: GraphEdge) -> None:
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            properties=edge.properties,
            source_segment_id=edge.source_segment_id
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        if not self._graph.has_node(node_id):
            return None
        
        data = self._graph.nodes[node_id]
        return GraphNode(
            id=node_id,
            type=data.get("type", "unknown"),
            label=data.get("label", node_id),
            description=data.get("description"),
            properties=data.get("properties", {}),
            source_segment_id=data.get("source_segment_id")
        )

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        if not self._graph.has_node(node_id):
            return []
        
        # Simple 1-hop for now, can expand for depth with nx.ego_graph
        neighbors = set(self._graph.neighbors(node_id))
        result = []
        for n_id in neighbors:
            node = self.get_node(n_id)
            if node:
                result.append(node)
        return result

    def search_nodes(self, query: str) -> List[GraphNode]:
        # Linear search - inefficient for large graphs but fine for demo
        results = []
        q = query.lower()
        for node_id, data in self._graph.nodes(data=True):
            label = data.get("label", "").lower()
            desc = data.get("description", "") or ""
            if q in label or q in desc.lower():
                results.append(self.get_node(node_id))
        return results

    def save_to_file(self, path: str):
        nx.write_gml(self._graph, path)
    
    def load_from_file(self, path: str):
        self._graph = nx.read_gml(path)
        
    # Async Implementation for NetworkX (Just wraps sync calls)
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
