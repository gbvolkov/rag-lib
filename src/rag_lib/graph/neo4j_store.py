from typing import List, Optional, Dict, Any
from rag_lib.graph.domain import GraphNode, GraphEdge
from rag_lib.graph.store import BaseGraphStore
from rag_lib.core.logger import logger

class Neo4jGraphStore(BaseGraphStore):
    """
    Graph Store implementation using Neo4j.
    Requires `pip install neo4j` or `pip install rag-lib[graph]`.
    """
    def __init__(self, uri: str, auth: tuple[str, str], database: str = "neo4j"):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Neo4j driver is not installed. Please install `rag-lib[graph]`.")
        
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database
        logger.info(f"Initialized Neo4jGraphStore connected to {uri}")

    def close(self):
        self.driver.close()

    def add_node(self, node: GraphNode) -> None:
        query = (
            "MERGE (n:Entity {id: $id}) "
            "SET n.label = $label, n.type = $type, n.description = $description, "
            "n.source_segment_id = $source_segment_id, n += $properties"
        )
        params = {
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "description": node.description,
            "source_segment_id": node.source_segment_id,
            "properties": node.properties
        }
        with self.driver.session(database=self.database) as session:
            session.run(query, params)

    def add_edge(self, edge: GraphEdge) -> None:
        # Dynamic relationship type is tricky in Cypher via parameters, usually requires string injection.
        # We sanitize the relation_type to be safe (alphanumeric + underscore).
        rel_type = "".join(x for x in edge.relation_type if x.isalnum() or x == "_").upper()
        
        query = (
            "MATCH (a:Entity {id: $source_id}) "
            "MATCH (b:Entity {id: $target_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.weight = $weight, r.source_segment_id = $source_segment_id, r += $properties"
        )
        params = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "weight": edge.weight,
            "source_segment_id": edge.source_segment_id,
            "properties": edge.properties
        }
        with self.driver.session(database=self.database) as session:
            session.run(query, params)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        query = "MATCH (n:Entity {id: $id}) RETURN n"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=node_id).single()
            if result:
                record = result["n"]
                # Neo4j node properties to dict
                props = dict(record)
                # Extract known fields
                return GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                )
        return None

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        # Cypher variable length path
        query = (
            f"MATCH (n:Entity {{id: $id}})-[*1..{depth}]-(m:Entity) "
            "RETURN DISTINCT m"
        )
        nodes = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=node_id)
            for record in result:
                props = dict(record["m"])
                nodes.append(GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                ))
        return nodes

    def search_nodes(self, query: str) -> List[GraphNode]:
        # Fulltext search or simple CONTAINS
        # Using simplified CONTAINS for now
        cql = (
            "MATCH (n:Entity) "
            "WHERE toLower(n.label) CONTAINS toLower($q) OR toLower(n.description) CONTAINS toLower($q) "
            "RETURN n LIMIT 10"
        )
        nodes = []
        with self.driver.session(database=self.database) as session:
            result = session.run(cql, q=query)
            for record in result:
                props = dict(record["n"])
                nodes.append(GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                ))
        return nodes
    async def aadd_node(self, node: GraphNode) -> None:
        query = (
            "MERGE (n:Entity {id: $id}) "
            "SET n.label = $label, n.type = $type, n.description = $description, "
            "n.source_segment_id = $source_segment_id, n += $properties"
        )
        params = {
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "description": node.description,
            "source_segment_id": node.source_segment_id,
            "properties": node.properties
        }
        async with self.driver.session(database=self.database) as session:
            await session.run(query, params)

    async def aadd_edge(self, edge: GraphEdge) -> None:
        rel_type = "".join(x for x in edge.relation_type if x.isalnum() or x == "_").upper()
        
        query = (
            "MATCH (a:Entity {id: $source_id}) "
            "MATCH (b:Entity {id: $target_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.weight = $weight, r.source_segment_id = $source_segment_id, r += $properties"
        )
        params = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "weight": edge.weight,
            "source_segment_id": edge.source_segment_id,
            "properties": edge.properties
        }
        async with self.driver.session(database=self.database) as session:
             await session.run(query, params)

    async def aget_node(self, node_id: str) -> Optional[GraphNode]:
        query = "MATCH (n:Entity {id: $id}) RETURN n"
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, id=node_id)
            record_obj = await result.single()
            if record_obj:
                record = record_obj["n"]
                props = dict(record)
                return GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                )
        return None

    async def aget_neighbors(self, node_id: str, depth: int = 1) -> List[GraphNode]:
        query = (
            f"MATCH (n:Entity {{id: $id}})-[*1..{depth}]-(m:Entity) "
            "RETURN DISTINCT m"
        )
        nodes = []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(query, id=node_id)
            async for record in result:
                 props = dict(record["m"])
                 nodes.append(GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                ))
        return nodes

    async def asearch_nodes(self, query: str) -> List[GraphNode]:
        cql = (
            "MATCH (n:Entity) "
            "WHERE toLower(n.label) CONTAINS toLower($q) OR toLower(n.description) CONTAINS toLower($q) "
            "RETURN n LIMIT 10"
        )
        nodes = []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(cql, q=query)
            async for record in result:
                props = dict(record["n"])
                nodes.append(GraphNode(
                    id=props.get("id"),
                    type=props.get("type", "unknown"),
                    label=props.get("label", props.get("id")),
                    description=props.get("description"),
                    source_segment_id=props.get("source_segment_id"),
                    properties={k:v for k,v in props.items() if k not in ["id", "type", "label", "description", "source_segment_id"]}
                ))
        return nodes
