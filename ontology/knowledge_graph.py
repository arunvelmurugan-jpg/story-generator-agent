"""
Knowledge Graph for PHTN.AI Sub-Agent Framework

In-memory knowledge graph implementation with:
- Node and edge management
- Graph queries (traversal, pattern matching)
- Relationship inference
- Export to standard formats (OWL, RDF, JSON-LD)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime

from .types import (
    ConceptType,
    RelationshipType,
    OntologyConcept,
    OntologyEntity,
    OntologyRelationship,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type,
            "properties": self.properties,
            "metadata": self.metadata,
        }


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relationship_type,
            "properties": self.properties,
            "weight": self.weight,
        }


@dataclass
class GraphQuery:
    """Query for the knowledge graph."""
    query_type: str
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    relationship_types: List[str] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    max_depth: int = 3
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 100


@dataclass
class GraphQueryResult:
    """Result of a knowledge graph query."""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "paths": self.paths,
            "total_count": self.total_count,
            "query_time_ms": self.query_time_ms,
        }


@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge graph."""
    enable_inference: bool = True
    max_nodes: int = 10000
    max_edges: int = 50000
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


class KnowledgeGraph:
    """
    In-memory knowledge graph implementation.
    
    Features:
    - Node and edge management
    - Graph traversal queries
    - Pattern matching
    - Relationship inference
    - Export to OWL/RDF/JSON-LD
    """
    
    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        """
        Initialize knowledge graph.
        
        Args:
            config: Graph configuration
        """
        self.config = config or KnowledgeGraphConfig()
        
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._node_type_index: Dict[str, Set[str]] = defaultdict(set)
        self._edge_type_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._edge_counter = 0
        
        logger.debug("KnowledgeGraph initialized")
    
    def add_node(self, node: GraphNode) -> bool:
        """
        Add a node to the graph.
        
        Args:
            node: Node to add
            
        Returns:
            True if added, False if already exists
        """
        if len(self._nodes) >= self.config.max_nodes:
            logger.warning("Maximum nodes reached")
            return False
        
        if node.id in self._nodes:
            self._nodes[node.id] = node
            return True
        
        self._nodes[node.id] = node
        self._node_type_index[node.node_type].add(node.id)
        
        return True
    
    def add_edge(self, edge: GraphEdge) -> bool:
        """
        Add an edge to the graph.
        
        Args:
            edge: Edge to add
            
        Returns:
            True if added, False if failed
        """
        if len(self._edges) >= self.config.max_edges:
            logger.warning("Maximum edges reached")
            return False
        
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            logger.warning(f"Source or target node not found for edge {edge.id}")
            return False
        
        if not edge.id:
            self._edge_counter += 1
            edge.id = f"edge-{self._edge_counter}"
        
        self._edges[edge.id] = edge
        self._adjacency[edge.source_id].add(edge.target_id)
        self._reverse_adjacency[edge.target_id].add(edge.source_id)
        self._edge_type_index[edge.relationship_type].add(edge.id)
        
        return True
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        node_ids = self._node_type_index.get(node_type, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]
    
    def get_edges_by_type(self, edge_type: str) -> List[GraphEdge]:
        """Get all edges of a specific type."""
        edge_ids = self._edge_type_index.get(edge_type, set())
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]
    
    def get_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
        relationship_type: Optional[str] = None,
    ) -> List[GraphNode]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Source node ID
            direction: "outgoing", "incoming", or "both"
            relationship_type: Optional filter by relationship type
            
        Returns:
            List of neighboring nodes
        """
        neighbors = set()
        
        if direction in ("outgoing", "both"):
            neighbors.update(self._adjacency.get(node_id, set()))
        
        if direction in ("incoming", "both"):
            neighbors.update(self._reverse_adjacency.get(node_id, set()))
        
        if relationship_type:
            filtered = set()
            for neighbor_id in neighbors:
                for edge in self._edges.values():
                    if (edge.source_id == node_id and edge.target_id == neighbor_id and
                        edge.relationship_type == relationship_type):
                        filtered.add(neighbor_id)
                    elif (edge.target_id == node_id and edge.source_id == neighbor_id and
                          edge.relationship_type == relationship_type):
                        filtered.add(neighbor_id)
            neighbors = filtered
        
        return [self._nodes[nid] for nid in neighbors if nid in self._nodes]
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.
        
        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path length
            
        Returns:
            List of node IDs in path, or None if not found
        """
        if start_id not in self._nodes or end_id not in self._nodes:
            return None
        
        if start_id == end_id:
            return [start_id]
        
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            for neighbor in self._adjacency.get(current, set()):
                if neighbor == end_id:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def query(self, query: GraphQuery) -> GraphQueryResult:
        """
        Execute a query on the knowledge graph.
        
        Args:
            query: Query specification
            
        Returns:
            Query result
        """
        import time
        start_time = time.time()
        
        result_nodes = []
        result_edges = []
        result_paths = []
        
        if query.query_type == "neighbors":
            if query.start_node:
                neighbors = self.get_neighbors(
                    query.start_node,
                    direction="both",
                    relationship_type=query.relationship_types[0] if query.relationship_types else None,
                )
                result_nodes = neighbors[:query.limit]
        
        elif query.query_type == "path":
            if query.start_node and query.end_node:
                path = self.find_path(query.start_node, query.end_node, query.max_depth)
                if path:
                    result_paths = [path]
                    result_nodes = [self._nodes[nid] for nid in path if nid in self._nodes]
        
        elif query.query_type == "subgraph":
            if query.start_node:
                visited = set()
                self._traverse_subgraph(
                    query.start_node,
                    query.max_depth,
                    visited,
                    result_nodes,
                    result_edges,
                    query.relationship_types,
                )
        
        elif query.query_type == "by_type":
            if query.node_types:
                for node_type in query.node_types:
                    result_nodes.extend(self.get_nodes_by_type(node_type))
            if query.relationship_types:
                for edge_type in query.relationship_types:
                    result_edges.extend(self.get_edges_by_type(edge_type))
        
        elif query.query_type == "all":
            result_nodes = list(self._nodes.values())[:query.limit]
            result_edges = list(self._edges.values())
        
        query_time = (time.time() - start_time) * 1000
        
        return GraphQueryResult(
            nodes=result_nodes[:query.limit],
            edges=result_edges,
            paths=result_paths,
            total_count=len(result_nodes),
            query_time_ms=query_time,
        )
    
    def _traverse_subgraph(
        self,
        node_id: str,
        depth: int,
        visited: Set[str],
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        relationship_types: Optional[List[str]] = None,
    ):
        """Traverse subgraph from a starting node."""
        if depth <= 0 or node_id in visited:
            return
        
        visited.add(node_id)
        
        if node_id in self._nodes:
            nodes.append(self._nodes[node_id])
        
        for neighbor_id in self._adjacency.get(node_id, set()):
            for edge in self._edges.values():
                if edge.source_id == node_id and edge.target_id == neighbor_id:
                    if not relationship_types or edge.relationship_type in relationship_types:
                        edges.append(edge)
                        self._traverse_subgraph(
                            neighbor_id, depth - 1, visited, nodes, edges, relationship_types
                        )
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        self._node_type_index[node.node_type].discard(node_id)
        
        edges_to_remove = []
        for edge_id, edge in self._edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)
        
        del self._nodes[node_id]
        self._adjacency.pop(node_id, None)
        self._reverse_adjacency.pop(node_id, None)
        
        return True
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        self._adjacency[edge.source_id].discard(edge.target_id)
        self._reverse_adjacency[edge.target_id].discard(edge.source_id)
        self._edge_type_index[edge.relationship_type].discard(edge_id)
        
        del self._edges[edge_id]
        return True
    
    def clear(self):
        """Clear all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        self._node_type_index.clear()
        self._edge_type_index.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": {k: len(v) for k, v in self._node_type_index.items()},
            "edge_types": {k: len(v) for k, v in self._edge_type_index.items()},
        }
    
    def export_json_ld(self) -> Dict[str, Any]:
        """Export graph as JSON-LD."""
        return {
            "@context": {
                "@vocab": "https://phtn.ai/ontology/",
                "nodes": "@graph",
            },
            "@graph": [
                {
                    "@id": node.id,
                    "@type": node.node_type,
                    "label": node.label,
                    **node.properties,
                }
                for node in self._nodes.values()
            ],
            "edges": [edge.to_dict() for edge in self._edges.values()],
        }
    
    def export_cytoscape(self) -> Dict[str, Any]:
        """Export graph in Cytoscape.js format."""
        return {
            "elements": {
                "nodes": [
                    {
                        "data": {
                            "id": node.id,
                            "label": node.label,
                            "type": node.node_type,
                            **node.properties,
                        }
                    }
                    for node in self._nodes.values()
                ],
                "edges": [
                    {
                        "data": {
                            "id": edge.id,
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "label": edge.relationship_type,
                        }
                    }
                    for edge in self._edges.values()
                ],
            }
        }
