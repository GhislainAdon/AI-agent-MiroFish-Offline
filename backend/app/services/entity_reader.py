"""
Service de lecture et de filtrage des entités.
Lit les nœuds depuis le graphe Neo4j, filtre les nœuds de type d'entité significatifs.

Remplace zep_entity_reader.py — tous les appels Zep Cloud sont remplacés par GraphStorage.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..storage import GraphStorage

logger = get_logger('mirofish.entity_reader')


@dataclass
class EntityNode:
    """Structure de données de nœud d'entité"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Arêtes associées
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Autres nœuds associés
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Obtenir le type d'entité (exclure le label Entity par défaut)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Ensemble d'entités filtrées"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    Service de lecture et de filtrage des entités (via GraphStorage / Neo4j)

    Capacités principales :
    1. Lire tous les nœuds depuis le graphe
    2. Filtrer les nœuds de type d'entité significatifs (nœuds dont les labels ne sont pas uniquement "Entity")
    3. Obtenir les arêtes associées et les informations des nœuds liés pour chaque entité
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Obtenir tous les nœuds du graphe.

        Args:
            graph_id: ID du graphe

        Returns:
            Liste de nœuds.
        """
        logger.info(f"Obtention de tous les nœuds du graphe {graph_id}...")
        nodes = self.storage.get_all_nodes(graph_id)
        logger.info(f"{len(nodes)} nœuds obtenus au total")
        return nodes

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Obtenir toutes les arêtes du graphe.

        Args:
            graph_id: ID du graphe

        Returns:
            Liste d'arêtes.
        """
        logger.info(f"Obtention de toutes les arêtes du graphe {graph_id}...")
        edges = self.storage.get_all_edges(graph_id)
        logger.info(f"{len(edges)} arêtes obtenues au total")
        return edges

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Obtenir toutes les arêtes associées à un nœud spécifié.

        Args:
            node_uuid: UUID du nœud

        Returns:
            Liste d'arêtes.
        """
        try:
            return self.storage.get_node_edges(node_uuid)
        except Exception as e:
            logger.warning(f"Échec de l'obtention des arêtes pour le nœud {node_uuid} : {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Filtrer et extraire les nœuds avec des types d'entité significatifs.

        Logique de filtrage :
        - Si les labels d'un nœud incluent uniquement "Entity", il n'a pas de type significatif et est ignoré.
        - Si les labels d'un nœud incluent d'autres labels que "Entity" et "Node", il a un type significatif et est conservé.

        Args:
            graph_id: ID du graphe
            defined_entity_types: Liste optionnelle de types d'entités à filtrer. Si fournie, seules les entités correspondant à l'un de ces types sont conservées.
            enrich_with_edges: Récupérer ou non les informations d'arêtes associées à chaque entité.

        Returns:
            FilteredEntities : Collection d'entités filtrées.
        """
        logger.info(f"Début du filtrage des entités dans le graphe {graph_id}...")

        # Obtenir tous les nœuds
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        # Obtenir toutes les arêtes (pour la recherche d'associations ultérieure)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        # Construire le mapping de l'UUID du nœud vers les données du nœud
        node_map = {n["uuid"]: n for n in all_nodes}

        # Filtrer les entités correspondant aux critères
        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # Logique de filtrage : les labels doivent contenir d'autres labels que "Entity" et "Node"
            custom_labels = [la for la in labels if la not in ["Entity", "Node"]]

            if not custom_labels:
                # Uniquement les labels par défaut, ignorer
                continue

            # Si des types prédéfinis sont spécifiés, vérifier la correspondance
            if defined_entity_types:
                matching_labels = [la for la in custom_labels if la in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            # Créer l'objet nœud d'entité
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            # Obtenir les arêtes et nœuds associés
            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                # Obtenir les nœuds liés associés avec leurs informations
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node.get("labels", []),
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"Filtrage terminé : total nœuds {total_count}, correspondants {len(filtered_entities)}, "
                     f"types d'entités : {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Obtenir une seule entité avec son contexte complet (arêtes et nœuds associés).

        Optimisé : utilise get_node() + get_node_edges() au lieu de charger TOUS les nœuds.
        Ne récupère les nœuds associés que individuellement selon les besoins.

        Args:
            graph_id: ID du graphe
            entity_uuid: UUID de l'entité

        Returns:
            EntityNode ou None.
        """
        try:
            # Obtenir le nœud directement par UUID (recherche O(1))
            node = self.storage.get_node(entity_uuid)
            if not node:
                return None

            # Obtenir les arêtes pour ce nœud (O(degré) via Cypher)
            edges = self.storage.get_node_edges(entity_uuid)

            # Traiter les arêtes associées et collecter les UUID des nœuds associés
            related_edges = []
            related_node_uuids: Set[str] = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # Récupérer les nœuds associés individuellement (évite de charger TOUS les nœuds)
            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = self.storage.get_node(related_uuid)
                if related_node:
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node.get("labels", []),
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Échec de l'obtention de l'entité {entity_uuid} : {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Obtenir toutes les entités d'un type spécifié.

        Args:
            graph_id: ID du graphe
            entity_type: Type d'entité (ex. "Student", "PublicFigure", etc.)
            enrich_with_edges: Récupérer ou non les informations d'arêtes associées à chaque entité.

        Returns:
            Liste des entités du type spécifié.
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
