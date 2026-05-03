"""
GraphStorage — interface abstraite pour les backends de stockage de graphes.

Tous les appels Zep Cloud sont remplacés par cette abstraction.
Implémentation actuelle : Neo4jStorage (neo4j_storage.py).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable


class GraphStorage(ABC):
    """Interface abstraite pour les backends de stockage de graphes."""

    # --- Cycle de vie du graphe ---

    @abstractmethod
    def create_graph(self, name: str, description: str = "") -> str:
        """Créer un nouveau graphe. Retourne graph_id."""

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """Supprimer un graphe et tous ses nœuds/arêtes."""

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        """Stocker l'ontologie (types d'entités + types de relations) pour un graphe."""

    @abstractmethod
    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        """Récupérer l'ontologie stockée pour un graphe."""

    # --- Ajout de données ---

    @abstractmethod
    def add_text(self, graph_id: str, text: str) -> str:
        """
        Traiter le texte : NER/RE → créer nœuds/arêtes → retourner episode_id.
        Ceci est synchrone (contrairement aux épisodes asynchrones de Zep Cloud).
        """

    @abstractmethod
    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Ajouter des morceaux de texte par lot. Retourne une liste d'episode_ids."""

    @abstractmethod
    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """
        Attendre que les épisodes soient traités.
        Pour Neo4j : sans opération (traitement synchrone).
        Conservé pour la compatibilité API avec les appelants de l'ère Zep.
        """

    # --- Lecture des nœuds ---

    @abstractmethod
    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """Obtenir tous les nœuds d'un graphe (avec une limite facultative)."""

    @abstractmethod
    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Obtenir un seul nœud par UUID."""

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Obtenir toutes les arêtes connectées à un nœud (O(1) via Cypher, pas de balayage complet)."""

    @abstractmethod
    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        """Obtenir les nœuds filtrés par le label de type d'entité."""

    # --- Lecture des arêtes ---

    @abstractmethod
    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Obtenir toutes les arêtes d'un graphe."""

    # --- Recherche ---

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        Recherche hybride (vectorielle + par mots-clés) sur les données du graphe.

        Args:
            graph_id: Graphe dans lequel rechercher
            query: Texte de la requête de recherche
            limit: Nombre maximal de résultats
            scope: "edges", "nodes", ou "both"

        Returns:
            Dictionnaire avec des listes 'edges' et/ou 'nodes' (encapsulées par GraphToolsService dans SearchResult)
        """

    # --- Informations sur le graphe ---

    @abstractmethod
    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        """Obtenir les métadonnées du graphe (nombre de nœuds, nombre d'arêtes, types d'entités)."""

    @abstractmethod
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Obtenir les données complètes du graphe (format enrichi pour le frontend).

        Retourne un dictionnaire avec :
            graph_id, nodes, edges, node_count, edge_count
        Les dictionnaires d'arêtes incluent les champs dérivés : fact_type, source_node_name, target_node_name
        """
