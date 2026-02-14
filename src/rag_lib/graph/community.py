from typing import Dict, List, Any
import networkx as nx
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.core.logger import logger

class CommunityDetector:
    """
    Detects communities in the graph using NetworkX algorithms.
    """
    
    @staticmethod
    def detect(store: NetworkXGraphStore) -> Dict[int, List[str]]:
        """
        Detects communities using Clauset-Newman-Moore greedy modularity maximization.
        Returns a dictionary mapping community_id to a list of node_ids.
        
        Args:
            store: The NetworkXGraphStore instance containing the graph.
            
        Returns:
            Dict[int, List[str]]: {community_id: [node_id1, node_id2, ...]}
        """
        if not isinstance(store, NetworkXGraphStore):
            logger.warning("Community detection currently only supports NetworkXGraphStore.")
            return {}
            
        graph = store.graph
        if graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, no communities to detect.")
            return {}

        logger.info(f"Detecting communities for graph with {graph.number_of_nodes()} nodes...")
        
        # Use greedy_modularity_communities (fast, reasonable quality)
        try:
            communities = nx.community.greedy_modularity_communities(graph)
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
        
        # Format result
        result = {}
        for i, community in enumerate(communities):
            # community is a frozenset of nodes
            result[i] = list(community)
            
        logger.info(f"Detected {len(result)} communities.")
        return result
