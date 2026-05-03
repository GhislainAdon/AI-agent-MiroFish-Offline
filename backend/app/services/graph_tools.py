"""
Service d'outils de recherche dans le graphe
Encapsule la recherche dans le graphe, la récupération de nœuds, les requêtes d'arêtes et d'autres outils pour le Report Agent.

Remplace zep_tools.py — tous les appels Zep Cloud remplacés par GraphStorage.

Outils de recherche principaux (optimisés) :
1. InsightForge (Recherche approfondie) - Recherche hybride la plus puissante, génère automatiquement des sous-questions et effectue une recherche multidimensionnelle
2. PanoramaSearch (Recherche panoramique) - Obtenir une vue complète, y compris le contenu expiré
3. QuickSearch (Recherche simple) - Recherche rapide
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..storage import GraphStorage

logger = get_logger('mirofish.graph_tools')


@dataclass
class SearchResult:
    """Résultat de recherche"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }

    def to_text(self) -> str:
        """Convertir au format texte pour la compréhension du LLM"""
        text_parts = [f"Requête de recherche : {self.query}", f"{self.total_count} résultats connexes trouvés"]

        if self.facts:
            text_parts.append("\n### Faits connexes :")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Informations sur le nœud"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }

    def to_text(self) -> str:
        """Convertir au format texte"""
        entity_type = next((la for la in self.labels if la not in ["Entity", "Node"]), "Type inconnu")
        return f"Entité : {self.name} (Type : {entity_type})\nRésumé : {self.summary}"


@dataclass
class EdgeInfo:
    """Informations sur l'arête"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Information temporelle (peut être absente dans Neo4j — conservée pour la compatibilité de l'interface)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convertir au format texte"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Relation : {source} --[{self.name}]--> {target}\nFait : {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Inconnu"
            invalid_at = self.invalid_at or "Actuel"
            base_text += f"\nPlage temporelle : {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expiré : {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Si déjà expiré"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Si déjà invalide"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Résultat de recherche approfondie (InsightForge)
    Contient les résultats de recherche de plusieurs sous-questions et l'analyse intégrée
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Résultats de recherche par dimension
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)

    # Informations statistiques
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }

    def to_text(self) -> str:
        """Convertir au format texte détaillé pour la compréhension du LLM"""
        text_parts = [
            f"## Analyse approfondie des prédictions futures",
            f"Requête d'analyse : {self.query}",
            f"Scénario de prédiction : {self.simulation_requirement}",
            f"\n### Statistiques des données de prédiction",
            f"- Faits de prédiction connexes : {self.total_facts}",
            f"- Entités impliquées : {self.total_entities}",
            f"- Chaînes de relations : {self.total_relationships}"
        ]

        if self.sub_queries:
            text_parts.append(f"\n### Sous-questions d'analyse")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        if self.semantic_facts:
            text_parts.append(f"\n### Faits clés (veuillez les citer textuellement dans le rapport)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.entity_insights:
            text_parts.append(f"\n### Entités principales")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Inconnu')}** ({entity.get('type', 'Entity')})")
                if entity.get('summary'):
                    text_parts.append(f"  Résumé : \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Faits connexes : {len(entity.get('related_facts', []))} faits")

        if self.relationship_chains:
            text_parts.append(f"\n### Chaînes de relations")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Résultat de recherche panoramique (Panorama)
    Contient toutes les informations connexes, y compris le contenu expiré
    """
    query: str

    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)

    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }

    def to_text(self) -> str:
        """Convertir au format texte (version complète, sans troncature)"""
        text_parts = [
            f"## Résultats de recherche panoramique (Vue panoramique du futur)",
            f"Requête : {self.query}",
            f"\n### Statistiques",
            f"- Nœuds totaux : {self.total_nodes}",
            f"- Arêtes totales : {self.total_edges}",
            f"- Faits actuellement valides : {self.active_count}",
            f"- Faits historiques/expirés : {self.historical_count}"
        ]

        if self.active_facts:
            text_parts.append(f"\n### Faits actuellement valides (Résultats de simulation textuels)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.historical_facts:
            text_parts.append(f"\n### Faits historiques/expirés (Registre d'évolution)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        if self.all_nodes:
            text_parts.append(f"\n### Entités impliquées")
            for node in self.all_nodes:
                entity_type = next((la for la in node.labels if la not in ["Entity", "Node"]), "Entity")
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Résultat d'entretien d'un agent"""
    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Biographie : {self.agent_bio}_\n\n"
        text += f"**Q :** {self.question}\n\n"
        text += f"**R :** {self.response}\n"
        if self.key_quotes:
            text += "\n**Citations clés :**\n"
            for quote in self.key_quotes:
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Résultat d'entretien
    Contient les réponses d'entretien de plusieurs agents simulés
    """
    interview_topic: str
    interview_questions: List[str]

    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)

    selection_reasoning: str = ""
    summary: str = ""

    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }

    def to_text(self) -> str:
        """Convertir au format texte détaillé pour la compréhension du LLM et la référence du rapport"""
        text_parts = [
            "## Rapport d'entretien approfondi",
            f"**Sujet d'entretien :** {self.interview_topic}",
            f"**Personnes interrogées :** {self.interviewed_count} / {self.total_agents} Agents simulés",
            "\n### Justification de la sélection",
            self.selection_reasoning or "(Sélection automatique)",
            "\n---",
            "\n### Transcriptions des entretiens",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Entretien n°{i} : {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Aucun compte-rendu d'entretien)\n\n---")

        text_parts.append("\n### Résumé des entretiens et points clés")
        text_parts.append(self.summary or "(Aucun résumé)")

        return "\n".join(text_parts)


class GraphToolsService:
    """
    Service d'outils de recherche dans le graphe (via GraphStorage / Neo4j)

    [Outils de recherche principaux - Optimisés]
    1. insight_forge - Recherche approfondie (Le plus puissant, génère automatiquement des sous-questions, recherche multidimensionnelle)
    2. panorama_search - Recherche panoramique (Obtenir une vue complète, y compris le contenu expiré)
    3. quick_search - Recherche simple (Recherche rapide)
    4. interview_agents - Entretien approfondi (Interroger les agents simulés, obtenir des perspectives multiples)

    [Outils de base]
    - search_graph - Recherche sémantique dans le graphe
    - get_all_nodes - Obtenir tous les nœuds du graphe
    - get_all_edges - Obtenir toutes les arêtes du graphe (avec information temporelle)
    - get_node_detail - Obtenir des informations détaillées sur un nœud
    - get_node_edges - Obtenir les arêtes liées à un nœud
    - get_entities_by_type - Obtenir les entités par type
    - get_entity_summary - Obtenir le résumé des relations d'une entité
    """

    def __init__(self, storage: GraphStorage, llm_client: Optional[LLMClient] = None):
        self.storage = storage
        self._llm_client = llm_client
        logger.info("Initialisation de GraphToolsService terminée")

    @property
    def llm(self) -> LLMClient:
        """Initialisation paresseuse du client LLM"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ========== Outils de base ==========

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Recherche sémantique dans le graphe (hybride : vectoriel + BM25 via Neo4j)

        Args:
            graph_id: ID du graphe
            query: Requête de recherche
            limit: Nombre de résultats à retourner
            scope: Portée de la recherche, "edges" ou "nodes" ou "both"

        Returns:
            SearchResult
        """
        logger.info(f"Recherche dans le graphe : graph_id={graph_id}, requête={query[:50]}...")

        try:
            search_results = self.storage.search(
                graph_id=graph_id,
                query=query,
                limit=limit,
                scope=scope,
            )

            facts = []
            edges = []
            nodes = []

            # Analyser les résultats d'arêtes
            if hasattr(search_results, 'edges'):
                edge_list = search_results.edges
            elif isinstance(search_results, dict) and 'edges' in search_results:
                edge_list = search_results['edges']
            else:
                edge_list = []

            for edge in edge_list:
                if isinstance(edge, dict):
                    fact = edge.get('fact', '')
                    if fact:
                        facts.append(fact)
                    edges.append({
                        "uuid": edge.get('uuid', ''),
                        "name": edge.get('name', ''),
                        "fact": fact,
                        "source_node_uuid": edge.get('source_node_uuid', ''),
                        "target_node_uuid": edge.get('target_node_uuid', ''),
                    })

            # Analyser les résultats de nœuds
            if hasattr(search_results, 'nodes'):
                node_list = search_results.nodes
            elif isinstance(search_results, dict) and 'nodes' in search_results:
                node_list = search_results['nodes']
            else:
                node_list = []

            for node in node_list:
                if isinstance(node, dict):
                    nodes.append({
                        "uuid": node.get('uuid', ''),
                        "name": node.get('name', ''),
                        "labels": node.get('labels', []),
                        "summary": node.get('summary', ''),
                    })
                    summary = node.get('summary', '')
                    if summary:
                        facts.append(f"[{node.get('name', '')}]: {summary}")

            logger.info(f"Recherche terminée : {len(facts)} faits connexes trouvés")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )

        except Exception as e:
            logger.warning(f"Échec de la recherche dans le graphe, passage à la recherche locale : {str(e)}")
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Recherche locale par correspondance de mots-clés (approche de secours)
        """
        logger.info(f"Utilisation de la recherche locale : requête={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []

        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                all_edges = self.storage.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.get("fact", "")) + match_score(edge.get("name", ""))
                    if score > 0:
                        scored_edges.append((score, edge))

                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    fact = edge.get("fact", "")
                    if fact:
                        facts.append(fact)
                    edges_result.append({
                        "uuid": edge.get("uuid", ""),
                        "name": edge.get("name", ""),
                        "fact": fact,
                        "source_node_uuid": edge.get("source_node_uuid", ""),
                        "target_node_uuid": edge.get("target_node_uuid", ""),
                    })

            if scope in ["nodes", "both"]:
                all_nodes = self.storage.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.get("name", "")) + match_score(node.get("summary", ""))
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.get("uuid", ""),
                        "name": node.get("name", ""),
                        "labels": node.get("labels", []),
                        "summary": node.get("summary", ""),
                    })
                    summary = node.get("summary", "")
                    if summary:
                        facts.append(f"[{node.get('name', '')}]: {summary}")

            logger.info(f"Recherche locale terminée : {len(facts)} faits connexes trouvés")

        except Exception as e:
            logger.error(f"Échec de la recherche locale : {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Obtenir tous les nœuds du graphe"""
        logger.info(f"Obtention de tous les nœuds du graphe {graph_id}...")

        raw_nodes = self.storage.get_all_nodes(graph_id)

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            ))

        logger.info(f"{len(result)} nœuds récupérés")
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """Obtenir toutes les arêtes du graphe (avec information temporelle)"""
        logger.info(f"Obtention de toutes les arêtes du graphe {graph_id}...")

        raw_edges = self.storage.get_all_edges(graph_id)

        result = []
        for edge in raw_edges:
            edge_info = EdgeInfo(
                uuid=edge.get("uuid", ""),
                name=edge.get("name", ""),
                fact=edge.get("fact", ""),
                source_node_uuid=edge.get("source_node_uuid", ""),
                target_node_uuid=edge.get("target_node_uuid", "")
            )

            if include_temporal:
                edge_info.created_at = edge.get("created_at")
                edge_info.valid_at = edge.get("valid_at")
                edge_info.invalid_at = edge.get("invalid_at")
                edge_info.expired_at = edge.get("expired_at")

            result.append(edge_info)

        logger.info(f"{len(result)} arêtes récupérées")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Obtenir des informations détaillées sur un nœud"""
        logger.info(f"Obtention des détails du nœud : {node_uuid[:8]}...")

        try:
            node = self.storage.get_node(node_uuid)
            if not node:
                return None

            return NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            )
        except Exception as e:
            logger.error(f"Échec de l'obtention des détails du nœud : {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Obtenir toutes les arêtes liées à un nœud

        Optimisé : utilise storage.get_node_edges() (Cypher O(degré))
        au lieu de charger TOUTES les arêtes et de filtrer.
        """
        logger.info(f"Obtention des arêtes liées au nœud {node_uuid[:8]}...")

        try:
            raw_edges = self.storage.get_node_edges(node_uuid)

            result = []
            for edge in raw_edges:
                result.append(EdgeInfo(
                    uuid=edge.get("uuid", ""),
                    name=edge.get("name", ""),
                    fact=edge.get("fact", ""),
                    source_node_uuid=edge.get("source_node_uuid", ""),
                    target_node_uuid=edge.get("target_node_uuid", ""),
                    created_at=edge.get("created_at"),
                    valid_at=edge.get("valid_at"),
                    invalid_at=edge.get("invalid_at"),
                    expired_at=edge.get("expired_at"),
                ))

            logger.info(f"{len(result)} arêtes liées au nœud trouvées")
            return result

        except Exception as e:
            logger.warning(f"Échec de l'obtention des arêtes du nœud : {str(e)}")
            return []

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str
    ) -> List[NodeInfo]:
        """Obtenir les entités par type"""
        logger.info(f"Obtention des entités de type {entity_type}...")

        # Utiliser la requête optimisée par label depuis le stockage
        raw_nodes = self.storage.get_nodes_by_label(graph_id, entity_type)

        result = []
        for node in raw_nodes:
            result.append(NodeInfo(
                uuid=node.get("uuid", ""),
                name=node.get("name", ""),
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {})
            ))

        logger.info(f"{len(result)} entités de type {entity_type} trouvées")
        return result

    def get_entity_summary(
        self,
        graph_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """Obtenir le résumé des relations d'une entité spécifique"""
        logger.info(f"Obtention du résumé des relations pour l'entité {entity_name}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )

        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Obtenir les statistiques du graphe"""
        logger.info(f"Obtention des statistiques du graphe {graph_id}...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }

    def get_simulation_context(
        self,
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Obtenir les informations contextuelles liées à la simulation"""
        logger.info(f"Obtention du contexte de simulation : {simulation_requirement[:50]}...")

        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )

        stats = self.get_graph_statistics(graph_id)

        all_nodes = self.get_all_nodes(graph_id)

        entities = []
        for node in all_nodes:
            custom_labels = [la for la in node.labels if la not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities)
        }

    # ========== Outils de recherche principaux (Optimisés) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        [InsightForge - Recherche approfondie]

        La fonction de recherche hybride la plus puissante, décompose automatiquement les problèmes et effectue une recherche multidimensionnelle :
        1. Utiliser le LLM pour décomposer le problème en plusieurs sous-questions
        2. Effectuer une recherche sémantique sur chaque sous-question
        3. Extraire les entités connexes et obtenir leurs informations détaillées
        4. Tracer les chaînes de relations
        5. Intégrer tous les résultats et générer des insights approfondis
        """
        logger.info(f"Recherche approfondie InsightForge : {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )

        # Étape 1 : Utiliser le LLM pour générer des sous-questions
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(f"{len(sub_queries)} sous-questions générées")

        # Étape 2 : Effectuer une recherche sémantique sur chaque sous-question
        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        # Rechercher également la question originale
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Étape 3 : Extraire les UUID des entités connexes depuis les arêtes
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # Obtenir les détails des entités connexes
        entity_insights = []
        node_map = {}

        for uuid in list(entity_uuids):
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((la for la in node.labels if la not in ["Entity", "Node"]), "Entity")

                    related_facts = [
                        f for f in all_facts
                        if node.name.lower() in f.lower()
                    ]

                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts
                    })
            except Exception as e:
                logger.debug(f"Échec de l'obtention du nœud {uuid} : {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Étape 4 : Construire les chaînes de relations
        relationship_chains = []
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')

                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(f"InsightForge terminé : {result.total_facts} faits, {result.total_entities} entités, {result.total_relationships} relations")
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """Utiliser le LLM pour générer des sous-questions"""
        system_prompt = """Vous êtes un expert professionnel en analyse de questions. Votre tâche est de décomposer une question complexe en plusieurs sous-questions pouvant être observées indépendamment dans un monde simulé.

Exigences :
1. Chaque sous-question doit être suffisamment spécifique pour trouver des comportements ou événements d'agent connexes dans le monde simulé
2. Les sous-questions doivent couvrir différentes dimensions de la question originale (par ex., qui, quoi, pourquoi, comment, quand, où)
3. Les sous-questions doivent être pertinentes par rapport au scénario de simulation
4. Retourner au format JSON : {"sub_queries": ["sous-question 1", "sous-question 2", ...]}"""

        user_prompt = f"""Contexte de l'exigence de simulation :
{simulation_requirement}

{f"Contexte du rapport : {report_context[:500]}" if report_context else ""}

Veuillez décomposer la question suivante en {max_queries} sous-questions :
{query}

Retourner les sous-questions sous forme de liste JSON."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(f"Échec de la génération des sous-questions : {str(e)}, utilisation des sous-questions par défaut")
            return [
                query,
                f"Principaux participants à {query}",
                f"Causes et impacts de {query}",
                f"Processus de développement de {query}"
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        [PanoramaSearch - Recherche panoramique]

        Obtenir une vue panoramique complète, y compris tout le contenu connexe et les informations historiques/expirées.
        """
        logger.info(f"Recherche panoramique PanoramaSearch : {query[:50]}...")

        result = PanoramaResult(query=query)

        # Obtenir tous les nœuds
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Obtenir toutes les arêtes (y compris les informations temporelles)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Catégoriser les faits
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]

            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                valid_at = edge.valid_at or "Inconnu"
                invalid_at = edge.invalid_at or edge.expired_at or "Inconnu"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                active_facts.append(edge.fact)

        # Trier par pertinence en fonction de la requête
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(f"PanoramaSearch terminé : {result.active_count} valides, {result.historical_count} historiques")
        return result

    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        [QuickSearch - Recherche simple]
        Outil de recherche rapide et léger.
        """
        logger.info(f"Recherche simple QuickSearch : {query[:50]}...")

        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )

        logger.info(f"QuickSearch terminé : {result.total_count} résultats")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        [InterviewAgents - Entretien approfondi]

        Appelle la véritable API d'entretien OASIS pour interroger les agents en cours de simulation.
        Cette méthode n'utilise PAS GraphStorage — elle appelle SimulationRunner
        et lit les profils d'agents depuis le disque.
        """
        from .simulation_runner import SimulationRunner

        logger.info(f"Entretien approfondi InterviewAgents (API réelle) : {interview_requirement[:50]}...")

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )

        # Étape 1 : Lire les fichiers de profils d'agents
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"Aucun fichier de profil trouvé pour la simulation {simulation_id}")
            result.summary = "Aucun fichier de profil d'agent trouvé pour l'entretien"
            return result

        result.total_agents = len(profiles)
        logger.info(f"{len(profiles)} profils d'agents chargés")

        # Étape 2 : Utiliser le LLM pour sélectionner les agents à interroger
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(f"{len(selected_agents)} agents sélectionnés pour l'entretien : {selected_indices}")

        # Étape 3 : Générer les questions d'entretien
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(f"{len(result.interview_questions)} questions d'entretien générées")

        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])

        INTERVIEW_PROMPT_PREFIX = (
            "Vous êtes interviewé. Veuillez combiner votre profil de personnage, tous les souvenirs et actions passés, "
            "et répondre directement aux questions suivantes en texte brut.\n"
            "Exigences de réponse :\n"
            "1. Répondez directement en langage naturel, n'appelle aucun outil\n"
            "2. Ne retournez pas au format JSON ou au format d'appel d'outil\n"
            "3. N'utilisez pas de titres Markdown (par ex., #, ##, ###)\n"
            "4. Répondez aux questions dans l'ordre, chaque réponse commençant par 'Question X :' (X est le numéro de la question)\n"
            "5. Séparez chaque réponse par une ligne vide\n"
            "6. Fournissez des réponses substantielles, au moins 2-3 phrases par question\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Étape 4 : Appeler la véritable API d'entretien
        try:
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt
                })

            logger.info(f"Appel de l'API d'entretien par lot (double plateforme) : {len(interviews_request)} agents")

            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,
                timeout=180.0
            )

            logger.info(f"API d'entretien retournée : {api_result.get('interviews_count', 0)} résultats, succès={api_result.get('success')}")

            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Erreur inconnue")
                logger.warning(f"Échec de l'appel à l'API d'entretien : {error_msg}")
                result.summary = f"Échec de l'appel à l'API d'entretien : {error_msg}. Veuillez vérifier l'état de l'environnement de simulation OASIS."
                return result

            # Étape 5 : Analyser la réponse de l'API
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Inconnu")
                agent_bio = agent.get("bio", "")

                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                twitter_text = twitter_response if twitter_response else "(Aucune réponse de cette plateforme)"
                reddit_text = reddit_response if reddit_response else "(Aucune réponse de cette plateforme)"
                response_text = f"[Réponse plateforme Twitter]\n{twitter_text}\n\n[Réponse plateforme Reddit]\n{reddit_text}"

                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'Question\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', 'Question'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]

                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            logger.warning(f"Échec de l'appel à l'API d'entretien (environnement non lancé ?) : {e}")
            result.summary = f"Entretien échoué : {str(e)}. L'environnement de simulation est peut-être fermé. Veuillez vous assurer que l'environnement OASIS est en cours d'exécution."
            return result
        except Exception as e:
            logger.error(f"Exception lors de l'appel à l'API d'entretien : {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Une erreur s'est produite lors du processus d'entretien : {str(e)}"
            return result

        # Étape 6 : Générer le résumé de l'entretien
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )

        logger.info(f"InterviewAgents terminé : {result.interviewed_count} agents interrogés (double plateforme)")
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Nettoyer les encapsulations d'appels d'outil JSON dans les réponses des agents et extraire le contenu réel"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Charger les fichiers de profils d'agents pour la simulation"""
        import os
        import csv

        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )

        profiles = []

        # Essayer de lire le format JSON Reddit en priorité
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(f"{len(profiles)} profils chargés depuis reddit_profiles.json")
                return profiles
            except Exception as e:
                logger.warning(f"Échec de la lecture de reddit_profiles.json : {e}")

        # Essayer de lire le format CSV Twitter
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Inconnu"
                        })
                logger.info(f"{len(profiles)} profils chargés depuis twitter_profiles.csv")
                return profiles
            except Exception as e:
                logger.warning(f"Échec de la lecture de twitter_profiles.csv : {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """Utiliser le LLM pour sélectionner les agents à interroger"""

        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Inconnu"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)

        system_prompt = """Vous êtes un expert professionnel en planification d'entretiens. Votre tâche est de sélectionner les agents les plus appropriés pour l'entretien à partir de la liste d'agents simulés, en fonction des exigences de l'entretien.

Critères de sélection :
1. L'identité/profession de l'agent est pertinente par rapport au sujet de l'entretien
2. L'agent peut détenir des perspectives uniques ou précieuses
3. Sélectionner des perspectives diversifiées (par ex., partisans, opposants, neutres, experts, etc.)
4. Prioriser les rôles directement liés à l'événement

Retourner au format JSON :
{
    "selected_indices": [Liste des indices des agents sélectionnés],
    "reasoning": "Explication de la justification de la sélection"
}"""

        user_prompt = f"""Exigence de l'entretien :
{interview_requirement}

Contexte de simulation :
{simulation_requirement if simulation_requirement else "Non fourni"}

Liste des agents disponibles ({len(agent_summaries)} au total) :
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Veuillez sélectionner jusqu'à {max_agents} agents les plus appropriés pour l'entretien et expliquer votre justification de sélection."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Sélectionnés automatiquement en fonction de la pertinence")

            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(f"Échec de la sélection d'agents par LLM, utilisation de la sélection par défaut : {e}")
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Utilisation de la stratégie de sélection par défaut"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Utiliser le LLM pour générer des questions d'entretien"""

        agent_roles = [a.get("profession", "Inconnu") for a in selected_agents]

        system_prompt = """Vous êtes un journaliste/interviewer professionnel. Sur la base des exigences de l'entretien, générez 3 à 5 questions d'entretien approfondies.

Exigences pour les questions :
1. Questions ouvertes qui encouragent des réponses détaillées
2. Questions pouvant avoir des réponses différentes selon les rôles
3. Couvrir plusieurs dimensions : faits, points de vue, sentiments, etc.
4. Langage naturel, comme de véritables entretiens
5. Chaque question doit contenir moins de 50 caractères, concise et claire
6. Posez directement la question, n'incluez pas d'explication contextuelle ou de préfixe

Retourner au format JSON : {"questions": ["question1", "question2", ...]}"""

        user_prompt = f"""Exigence de l'entretien : {interview_requirement}

Contexte de simulation : {simulation_requirement if simulation_requirement else "Non fourni"}

Rôles des sujets interviewés : {', '.join(agent_roles)}

Veuillez générer 3 à 5 questions d'entretien."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )

            return response.get("questions", [f"Quel est votre point de vue sur {interview_requirement} ?"])

        except Exception as e:
            logger.warning(f"Échec de la génération des questions d'entretien : {e}")
            return [
                f"Quel est votre point de vue sur {interview_requirement} ?",
                "Quel impact cela a-t-il sur vous ou le groupe que vous représentez ?",
                "Comment pensez-vous que ce problème devrait être résolu ou amélioré ?"
            ]

    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Générer le résumé de l'entretien"""

        if not interviews:
            return "Aucun entretien terminé"

        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"[{interview.agent_name} ({interview.agent_role})]\n{interview.response[:500]}")

        system_prompt = """Vous êtes un rédacteur en chef professionnel. Veuillez générer un résumé d'entretien basé sur les réponses de plusieurs personnes interrogées.

Exigences du résumé :
1. Extraire les points de vue principaux de toutes les parties
2. Souligner le consensus et les divergences entre les points de vue
3. Mettre en évidence les citations remarquables
4. Rester objectif et neutre, ne favoriser aucun côté
5. Garder le résumé sous 1000 mots

Contraintes de format (à respecter) :
- Utiliser des paragraphes en texte brut, séparés par des lignes vides
- Ne pas utiliser de titres Markdown (par ex., #, ##, ###)
- Ne pas utiliser de séparateurs (par ex., ---, ***)
- Utiliser des guillemets appropriés lors de citations des interviewés
- Peut utiliser **gras** pour marquer les mots-clés, mais ne pas utiliser d'autre syntaxe Markdown"""

        user_prompt = f"""Sujet de l'entretien : {interview_requirement}

Contenu des entretiens :
{"".join(interview_texts)}

Veuillez générer un résumé des entretiens."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary

        except Exception as e:
            logger.warning(f"Échec de la génération du résumé d'entretien : {e}")
            return f"{len(interviews)} personnes interrogées, dont : " + ", ".join([i.agent_name for i in interviews])
