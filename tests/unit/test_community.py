import pytest
import networkx as nx
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.community import CommunityDetector

def test_community_detection_simple():
    store = NetworkXGraphStore()
    
    # Create two disconnected clusters (communities)
    # Cluster 1: A-B-C
    for n in ["A", "B", "C"]:
        store.add_node(GraphNode(id=n, type="Type", label=n))
    
    store.add_edge(GraphEdge(source_id="A", target_id="B", relation_type="REL"))
    store.add_edge(GraphEdge(source_id="B", target_id="C", relation_type="REL"))
    store.add_edge(GraphEdge(source_id="C", target_id="A", relation_type="REL"))
    
    # Cluster 2: X-Y-Z
    for n in ["X", "Y", "Z"]:
        store.add_node(GraphNode(id=n, type="Type", label=n))
    
    store.add_edge(GraphEdge(source_id="X", target_id="Y", relation_type="REL"))
    store.add_edge(GraphEdge(source_id="Y", target_id="Z", relation_type="REL"))
    store.add_edge(GraphEdge(source_id="Z", target_id="X", relation_type="REL"))
    
    # Detect
    communities = CommunityDetector.detect(store)
    
    # Should find at least 2 communities
    assert len(communities) >= 2
    
    # Flatten values to check membership
    seen_nodes = set()
    for nodes in communities.values():
        seen_nodes.update(nodes)
        
    assert "A" in seen_nodes
    assert "X" in seen_nodes
    
    # Check that A and X are likely in different communities
    # Find community for A
    comm_a = None
    for cid, nodes in communities.items():
        if "A" in nodes:
            comm_a = cid
            break
            
    # Check X is not in comm_a (unless resolution matched them, but highly unlikely for disconnected components)
    if comm_a is not None:
        assert "X" not in communities[comm_a]
