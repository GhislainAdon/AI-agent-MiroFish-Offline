"""
Neo4jStorage — implémentation Neo4j Community Edition de GraphStorage.

Remplace tous les appels API Zep Cloud par des requêtes Cypher Neo4j locales.
Inclut : CRUD, ingestion de texte basée sur NER/RE, recherche hybride, logique de nouvelle tentative.
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from neo4j import GraphDatabase, Session as Neo4jSession
from neo4j.exceptions import (
    TransientError,
    ServiceUnavailable,
    SessionExpired,
)

from ..config import Config
from .graph_storage import GraphStorage
from .embedding_service import EmbeddingService
from .ner_extractor import NERExtractor
from .search_service import SearchService
from . import neo4j_schema

logger = logging.getLogger('mirofish.neo4j_storage')


class Neo4jStorage(GraphStorage):
    """Implémentation Neo4j CE de l'interface GraphStorage."""

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1  # secondes

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        ner_extractor: Optional[NERExtractor] = None,
    ):
        self._uri = uri or Config.NEO4J_URI
        self._user = user or Config.NEO4J_USER
        self._password = password or Config.NEO4J_PASSWORD

        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._embedding = embedding_service or EmbeddingService()
        self._ner = ner_extractor or NERExtractor()
        self._search = SearchService(self._embedding)

        # Initialiser le schéma (index, contraintes)
        self._ensure_schema()

    def close(self):
        """Fermer la connexion du pilote Neo4j."""
        self._driver.close()

    def _ensure_schema(self):
        """Créer les index et contraintes s'ils n'existent pas."""
        with self._driver.session() as session:
            for query in neo4j_schema.ALL_SCHEMA_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"Avertissement de requête de schéma (peut déjà exister) : {e}")

    # ----------------------------------------------------------------
    # Enveloppe de nouvelle tentative
    # ----------------------------------------------------------------

    def _call_with_retry(self, func, *args, **kwargs):
        """
        Exécuter une fonction avec nouvelle tentative sur les erreurs transitoires Neo4j.
        Remplace 3 modèles de nouvelle tentative différents du codebase Zep.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                last_error = e
                wait = self.RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    f"Erreur transitoire Neo4j (tentative {attempt + 1}/{self.MAX_RETRIES}), "
                    f"nouvelle tentative dans {wait}s : {e}"
                )
                time.sleep(wait)
            except Exception:
                raise

        raise last_error  # type: ignore

    # ----------------------------------------------------------------
    # Cycle de vie du graphe
    # ----------------------------------------------------------------

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _create(tx):
            tx.run(
                """
                CREATE (g:Graph {
                    graph_id: $graph_id,
                    name: $name,
                    description: $description,
                    ontology_json: '{}',
                    created_at: $created_at
                })
                """,
                graph_id=graph_id,
                name=name,
                description=description,
                created_at=now,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _create)

        logger.info(f"Graphe '{name}' créé avec l'identifiant {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        def _delete(tx):
            # Supprimer toutes les entités et leurs relations
            tx.run(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                gid=graph_id,
            )
            # Supprimer le nœud graphe
            tx.run(
                "MATCH (g:Graph {graph_id: $gid}) DELETE g",
                gid=graph_id,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _delete)
        logger.info(f"Graphe {graph_id} supprimé")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        def _set(tx):
            tx.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                SET g.ontology_json = $ontology_json
                """,
                gid=graph_id,
                ontology_json=json.dumps(ontology, ensure_ascii=False),
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _set)

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (g:Graph {graph_id: $gid}) RETURN g.ontology_json AS oj",
                gid=graph_id,
            )
            record = result.single()
            if record and record["oj"]:
                return json.loads(record["oj"])
            return {}

    # ----------------------------------------------------------------
    # Ajout de données (NER → nœuds/arêtes)
    # ----------------------------------------------------------------

    def add_text(self, graph_id: str, text: str) -> str:
        """Traiter le texte : NER/RE → incorporation par lot → créer nœuds/arêtes → retourner episode_id."""
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Obtenir l'ontologie pour guider le NER
        ontology = self.get_ontology(graph_id)

        # Extraire les entités et les relations
        logger.info(f"[add_text] Début de l'extraction NER pour le morceau ({len(text)} caractères)...")
        extraction = self._ner.extract(text, ontology)
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        logger.info(
            f"[add_text] NER terminé : {len(entities)} entités, {len(relations)} relations"
        )

        # --- Incorporer tous les textes par lot en une seule fois ---
        entity_summaries = [f"{e['name']} ({e['type']})" for e in entities]
        fact_texts = [r.get("fact", f"{r['source']} {r['type']} {r['target']}") for r in relations]
        all_texts_to_embed = entity_summaries + fact_texts

        all_embeddings: list = []
        if all_texts_to_embed:
            logger.info(f"[add_text] Incorporation par lot de {len(all_texts_to_embed)} textes...")
            try:
                all_embeddings = self._embedding.embed_batch(all_texts_to_embed)
            except Exception as e:
                logger.warning(f"[add_text] L'incorporation par lot a échoué, repli vers vide : {e}")
                all_embeddings = [[] for _ in all_texts_to_embed]

        entity_embeddings = all_embeddings[:len(entities)]
        relation_embeddings = all_embeddings[len(entities):]
        logger.info(f"[add_text] Incorporation terminée, écriture dans Neo4j...")

        with self._driver.session() as session:
            # Créer le nœud épisode
            def _create_episode(tx):
                tx.run(
                    """
                    CREATE (ep:Episode {
                        uuid: $uuid,
                        graph_id: $graph_id,
                        data: $data,
                        processed: true,
                        created_at: $created_at
                    })
                    """,
                    uuid=episode_id,
                    graph_id=graph_id,
                    data=text,
                    created_at=now,
                )

            self._call_with_retry(session.execute_write, _create_episode)

            # FUSIONNER les entités (upsert par graph_id + nom + label principal)
            entity_uuid_map: Dict[str, str] = {}  # nom_minuscule -> uuid
            for idx, entity in enumerate(entities):
                ename = entity["name"]
                etype = entity["type"]
                attrs = entity.get("attributes", {})
                summary_text = entity_summaries[idx]
                embedding = entity_embeddings[idx] if idx < len(entity_embeddings) else []

                e_uuid = str(uuid.uuid4())
                entity_uuid_map[ename.lower()] = e_uuid

                def _merge_entity(tx, _uuid=e_uuid, _name=ename, _type=etype,
                                  _attrs=attrs, _embedding=embedding,
                                  _summary=summary_text, _now=now):
                    # FUSIONNER par graph_id + nom en minuscules pour dédupliquer
                    result = tx.run(
                        """
                        MERGE (n:Entity {graph_id: $gid, name_lower: $name_lower})
                        ON CREATE SET
                            n.uuid = $uuid,
                            n.name = $name,
                            n.summary = $summary,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding,
                            n.created_at = $now
                        ON MATCH SET
                            n.summary = CASE WHEN n.summary = '' OR n.summary IS NULL
                                THEN $summary ELSE n.summary END,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding
                        RETURN n.uuid AS uuid
                        """,
                        gid=graph_id,
                        name_lower=_name.lower(),
                        uuid=_uuid,
                        name=_name,
                        summary=_summary,
                        attrs_json=json.dumps(_attrs, ensure_ascii=False),
                        embedding=_embedding,
                        now=_now,
                    )
                    record = result.single()
                    return record["uuid"] if record else _uuid

                actual_uuid = self._call_with_retry(session.execute_write, _merge_entity)
                entity_uuid_map[ename.lower()] = actual_uuid

                # Ajouter le label de type d'entité
                if etype and etype != "Entity":
                    try:
                        def _add_label(tx, _name_lower=ename.lower()):
                            tx.run(
                                f"MATCH (n:Entity {{graph_id: $gid, name_lower: $nl}}) SET n:`{etype}`",
                                gid=graph_id,
                                nl=_name_lower,
                            )
                        self._call_with_retry(session.execute_write, _add_label)
                    except Exception as e:
                        logger.warning(f"Impossible d'ajouter le label '{etype}' à '{ename}' : {e}")

            # Créer les relations
            for idx, relation in enumerate(relations):
                source_name = relation["source"]
                target_name = relation["target"]
                rtype = relation["type"]
                fact = relation["fact"]

                source_uuid = entity_uuid_map.get(source_name.lower())
                target_uuid = entity_uuid_map.get(target_name.lower())

                if not source_uuid or not target_uuid:
                    logger.warning(
                        f"Ignorer la relation {source_name}->{target_name} : "
                        f"entité non trouvée dans les résultats d'extraction"
                    )
                    continue

                fact_embedding = relation_embeddings[idx] if idx < len(relation_embeddings) else []
                r_uuid = str(uuid.uuid4())

                def _create_relation(tx, _r_uuid=r_uuid, _source_uuid=source_uuid,
                                     _target_uuid=target_uuid, _rtype=rtype,
                                     _fact=fact, _fact_emb=fact_embedding,
                                     _episode_id=episode_id, _now=now):
                    tx.run(
                        """
                        MATCH (src:Entity {uuid: $src_uuid})
                        MATCH (tgt:Entity {uuid: $tgt_uuid})
                        CREATE (src)-[r:RELATION {
                            uuid: $uuid,
                            graph_id: $gid,
                            name: $name,
                            fact: $fact,
                            fact_embedding: $fact_embedding,
                            attributes_json: '{}',
                            episode_ids: [$episode_id],
                            created_at: $now,
                            valid_at: null,
                            invalid_at: null,
                            expired_at: null
                        }]->(tgt)
                        """,
                        src_uuid=_source_uuid,
                        tgt_uuid=_target_uuid,
                        uuid=_r_uuid,
                        gid=graph_id,
                        name=_rtype,
                        fact=_fact,
                        fact_embedding=_fact_emb,
                        episode_id=_episode_id,
                        now=_now,
                    )

                self._call_with_retry(session.execute_write, _create_relation)

        logger.info(f"[add_text] Morceau terminé : épisode={episode_id}")
        return episode_id

    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Ajouter des morceaux de texte par lot avec rapport de progression."""
        episode_ids = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
            episode_id = self.add_text(graph_id, chunk)
            episode_ids.append(episode_id)

            if progress_callback:
                progress = (i + 1) / total
                progress_callback(progress)

            logger.info(f"Morceau {i + 1}/{total} traité")

        return episode_ids

    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """Sans opération — le traitement est synchrone dans Neo4j."""
        if progress_callback:
            progress_callback(1.0)

    # ----------------------------------------------------------------
    # Lecture des nœuds
    # ----------------------------------------------------------------

    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                gid=graph_id,
                limit=limit,
            )
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n, labels(n) AS labels",
                uuid=uuid,
            )
            record = result.single()
            if record:
                return self._node_to_dict(record["n"], record["labels"])
            return None

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Cypher O(1) — PAS de balayage complet + filtre comme l'ancien code Zep."""
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {uuid: $uuid})-[r:RELATION]-(m:Entity)
                RETURN r, startNode(r).uuid AS src_uuid, endNode(r).uuid AS tgt_uuid
                """,
                uuid=node_uuid,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        def _read(tx):
            # Label dynamique dans la requête (sûr — le label vient de l'ontologie, pas de l'entrée utilisateur)
            query = f"""
                MATCH (n:Entity:`{label}` {{graph_id: $gid}})
                RETURN n, labels(n) AS labels
            """
            result = tx.run(query, gid=graph_id)
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Lecture des arêtes
    # ----------------------------------------------------------------

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid
                ORDER BY r.created_at DESC
                """,
                gid=graph_id,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Recherche
    # ----------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        Recherche hybride — retourne les résultats correspondant à la portée.

        Retourne un dictionnaire avec des listes 'edges' et/ou 'nodes'
        (les appelants comme zep_tools les encapsuleront dans SearchResult).
        """
        result = {"edges": [], "nodes": [], "query": query}

        with self._driver.session() as session:
            if scope in ("edges", "both"):
                result["edges"] = self._search.search_edges(
                    session, graph_id, query, limit
                )

            if scope in ("nodes", "both"):
                result["nodes"] = self._search.search_nodes(
                    session, graph_id, query, limit
                )

        return result

    # ----------------------------------------------------------------
    # Informations sur le graphe
    # ----------------------------------------------------------------

    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        def _read(tx):
            # Compter les nœuds
            node_result = tx.run(
                "MATCH (n:Entity {graph_id: $gid}) RETURN count(n) AS cnt",
                gid=graph_id,
            )
            node_count = node_result.single()["cnt"]

            # Compter les arêtes
            edge_result = tx.run(
                "MATCH ()-[r:RELATION {graph_id: $gid}]->() RETURN count(r) AS cnt",
                gid=graph_id,
            )
            edge_count = edge_result.single()["cnt"]

            # Types d'entités distincts
            label_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                UNWIND labels(n) AS lbl
                WITH lbl WHERE lbl <> 'Entity'
                RETURN DISTINCT lbl
                """,
                gid=graph_id,
            )
            entity_types = [record["lbl"] for record in label_result]

            return {
                "graph_id": graph_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": entity_types,
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Vidage complet du graphe avec format d'arête enrichi (pour le frontend).
        Inclut les champs dérivés : fact_type, source_node_name, target_node_name.
        """
        def _read(tx):
            # Obtenir tous les nœuds
            node_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                """,
                gid=graph_id,
            )
            nodes = []
            node_map: Dict[str, str] = {}  # uuid -> nom
            for record in node_result:
                nd = self._node_to_dict(record["n"], record["labels"])
                nodes.append(nd)
                node_map[nd["uuid"]] = nd["name"]

            # Obtenir toutes les arêtes avec les noms des nœuds source/cible (JOINTURE)
            edge_result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid,
                       src.name AS src_name, tgt.name AS tgt_name
                """,
                gid=graph_id,
            )
            edges = []
            for record in edge_result:
                ed = self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                # Champs enrichis pour le frontend
                ed["fact_type"] = ed["name"]
                ed["source_node_name"] = record["src_name"] or ""
                ed["target_node_name"] = record["tgt_name"] or ""
                # Alias hérité
                ed["episodes"] = ed.get("episode_ids", [])
                edges.append(ed)

            return {
                "graph_id": graph_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # Aides à la conversion en dictionnaire
    # ----------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node, labels: List[str]) -> Dict[str, Any]:
        """Convertir un nœud Neo4j au format de dictionnaire de nœud standard."""
        props = dict(node)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Supprimer les champs internes du dictionnaire
        props.pop("embedding", None)
        props.pop("name_lower", None)

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "labels": [l for l in labels if l != "Entity"] if labels else [],
            "summary": props.get("summary", ""),
            "attributes": attributes,
            "created_at": props.get("created_at"),
        }

    @staticmethod
    def _edge_to_dict(rel, source_uuid: str, target_uuid: str) -> Dict[str, Any]:
        """Convertir une relation Neo4j au format de dictionnaire d'arête standard."""
        props = dict(rel)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # Supprimer les champs internes
        props.pop("fact_embedding", None)

        episode_ids = props.get("episode_ids", [])
        if episode_ids and not isinstance(episode_ids, list):
            episode_ids = [str(episode_ids)]

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "fact": props.get("fact", ""),
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
            "attributes": attributes,
            "created_at": props.get("created_at"),
            "valid_at": props.get("valid_at"),
            "invalid_at": props.get("invalid_at"),
            "expired_at": props.get("expired_at"),
            "episode_ids": episode_ids,
        }
