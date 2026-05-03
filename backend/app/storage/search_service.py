"""
SearchService — recherche hybride (vectorielle + par mots-clés) sur les données de graphe Neo4j.

Remplace la recherche intégrée de Zep Cloud avec un reclasseur.
Score : 0.7 * score_vectoriel + 0.3 * score_mots-clés (BM25 via index plein texte).
"""

import logging
from typing import List, Dict, Any, Optional

from neo4j import Session as Neo4jSession

from .embedding_service import EmbeddingService

logger = logging.getLogger('mirofish.search')

# Cypher pour la recherche vectorielle sur les arêtes (faits)
_VECTOR_SEARCH_EDGES = """
CALL db.index.vector.queryRelationships('fact_embedding', $limit, $query_vector)
YIELD relationship, score
WHERE relationship.graph_id = $graph_id
RETURN relationship AS r, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher pour la recherche vectorielle sur les nœuds (entités)
_VECTOR_SEARCH_NODES = """
CALL db.index.vector.queryNodes('entity_embedding', $limit, $query_vector)
YIELD node, score
WHERE node.graph_id = $graph_id
RETURN node AS n, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher pour la recherche plein texte (BM25) sur les arêtes
_FULLTEXT_SEARCH_EDGES = """
CALL db.index.fulltext.queryRelationships('fact_fulltext', $query_text)
YIELD relationship, score
WHERE relationship.graph_id = $graph_id
RETURN relationship AS r, score
ORDER BY score DESC
LIMIT $limit
"""

# Cypher pour la recherche plein texte sur les nœuds
_FULLTEXT_SEARCH_NODES = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $query_text)
YIELD node, score
WHERE node.graph_id = $graph_id
RETURN node AS n, score
ORDER BY score DESC
LIMIT $limit
"""


class SearchService:
    """Recherche hybride combinant similarité vectorielle et correspondance par mots-clés."""

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding = embedding_service

    def search_edges(
        self,
        session: Neo4jSession,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rechercher des arêtes (faits/relations) en utilisant le score hybride.

        Retourne une liste de dictionnaires avec les propriétés de l'arête + 'score'.
        """
        query_vector = self.embedding.embed(query)

        # Recherche vectorielle
        vector_results = self._run_edge_vector_search(
            session, graph_id, query_vector, limit * 2
        )

        # Recherche par mots-clés
        keyword_results = self._run_edge_keyword_search(
            session, graph_id, query, limit * 2
        )

        # Fusionner et classer
        merged = self._merge_results(
            vector_results, keyword_results, key="uuid", limit=limit
        )
        return merged

    def search_nodes(
        self,
        session: Neo4jSession,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rechercher des nœuds (entités) en utilisant le score hybride.

        Retourne une liste de dictionnaires avec les propriétés du nœud + 'score'.
        """
        query_vector = self.embedding.embed(query)

        vector_results = self._run_node_vector_search(
            session, graph_id, query_vector, limit * 2
        )

        keyword_results = self._run_node_keyword_search(
            session, graph_id, query, limit * 2
        )

        merged = self._merge_results(
            vector_results, keyword_results, key="uuid", limit=limit
        )
        return merged

    def _run_edge_vector_search(
        self, session: Neo4jSession, graph_id: str, query_vector: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        """Exécuter une recherche de similarité vectorielle sur l'embedding de fait des arêtes."""
        try:
            result = session.run(
                _VECTOR_SEARCH_EDGES,
                graph_id=graph_id,
                query_vector=query_vector,
                limit=limit,
            )
            return [
                {**dict(record["r"]), "uuid": record["r"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"La recherche vectorielle d'arêtes a échoué (l'index peut ne pas exister encore) : {e}")
            return []

    def _run_edge_keyword_search(
        self, session: Neo4jSession, graph_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Exécuter une recherche plein texte (BM25) sur le fait et le nom de l'arête."""
        try:
            # Échapper les caractères spéciaux Lucene dans la requête
            safe_query = self._escape_lucene(query)
            result = session.run(
                _FULLTEXT_SEARCH_EDGES,
                graph_id=graph_id,
                query_text=safe_query,
                limit=limit,
            )
            return [
                {**dict(record["r"]), "uuid": record["r"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"La recherche par mots-clés d'arêtes a échoué : {e}")
            return []

    def _run_node_vector_search(
        self, session: Neo4jSession, graph_id: str, query_vector: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        """Exécuter une recherche de similarité vectorielle sur l'embedding d'entité."""
        try:
            result = session.run(
                _VECTOR_SEARCH_NODES,
                graph_id=graph_id,
                query_vector=query_vector,
                limit=limit,
            )
            return [
                {**dict(record["n"]), "uuid": record["n"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"La recherche vectorielle de nœuds a échoué : {e}")
            return []

    def _run_node_keyword_search(
        self, session: Neo4jSession, graph_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Exécuter une recherche plein texte sur le nom et le résumé de l'entité."""
        try:
            safe_query = self._escape_lucene(query)
            result = session.run(
                _FULLTEXT_SEARCH_NODES,
                graph_id=graph_id,
                query_text=safe_query,
                limit=limit,
            )
            return [
                {**dict(record["n"]), "uuid": record["n"]["uuid"], "_score": record["score"]}
                for record in result
            ]
        except Exception as e:
            logger.warning(f"La recherche par mots-clés de nœuds a échoué : {e}")
            return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        key: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fusionner les résultats vectoriels et par mots-clés avec un score pondéré.

        Normalise les scores dans la plage [0, 1] avant la combinaison.
        """
        # Normaliser les scores vectoriels
        v_max = max((r["_score"] for r in vector_results), default=1.0) or 1.0
        v_scores = {r[key]: r["_score"] / v_max for r in vector_results}

        # Normaliser les scores par mots-clés
        k_max = max((r["_score"] for r in keyword_results), default=1.0) or 1.0
        k_scores = {r[key]: r["_score"] / k_max for r in keyword_results}

        # Construire la carte de résultats combinés
        all_items: Dict[str, Dict[str, Any]] = {}
        for r in vector_results:
            all_items[r[key]] = {k: v for k, v in r.items() if k != "_score"}
        for r in keyword_results:
            if r[key] not in all_items:
                all_items[r[key]] = {k: v for k, v in r.items() if k != "_score"}

        # Calculer les scores hybrides
        scored = []
        for uid, item in all_items.items():
            v = v_scores.get(uid, 0.0)
            k = k_scores.get(uid, 0.0)
            combined = self.VECTOR_WEIGHT * v + self.KEYWORD_WEIGHT * k
            item["score"] = combined
            scored.append(item)

        # Trier par score combiné décroissant
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    @staticmethod
    def _escape_lucene(query: str) -> str:
        """Échapper les caractères spéciaux de requête Lucene."""
        special = r'+-&|!(){}[]^"~*?:\/'
        result = []
        for ch in query:
            if ch in special:
                result.append('\\')
            result.append(ch)
        return ''.join(result)
