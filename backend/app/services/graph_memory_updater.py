"""
Service de mise à jour de la mémoire du graphe qui traite les activités des agents et les met à jour dans le graphe Neo4j.

Remplace zep_graph_memory_updater.py — le client Zep est remplacé par GraphStorage.
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from ..config import Config
from ..utils.logger import get_logger
from ..storage import GraphStorage

logger = get_logger('mirofish.graph_memory_updater')


@dataclass
class AgentActivity:
    """Enregistrement d'activité d'agent"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """
        Convertir l'activité en description textuelle en langage naturel

        Utiliser le format de description en langage naturel pour que l'extracteur NER puisse extraire les entités et les relations
        """
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }

        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()

        return f"{self.agent_name} : {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"A publié une publication : \"{content}\""
        return "A publié une publication"

    def _describe_like_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f"A aimé la publication de {post_author} : \"{post_content}\""
        elif post_content:
            return f"A aimé une publication : \"{post_content}\""
        elif post_author:
            return f"A aimé une publication de {post_author}"
        return "A aimé une publication"

    def _describe_dislike_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f"N'a pas aimé la publication de {post_author} : \"{post_content}\""
        elif post_content:
            return f"N'a pas aimé une publication : \"{post_content}\""
        elif post_author:
            return f"N'a pas aimé une publication de {post_author}"
        return "N'a pas aimé une publication"

    def _describe_repost(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        if original_content and original_author:
            return f"A repartagé la publication de {original_author} : \"{original_content}\""
        elif original_content:
            return f"A repartagé une publication : \"{original_content}\""
        elif original_author:
            return f"A repartagé une publication de {original_author}"
        return "A repartagé une publication"

    def _describe_quote_post(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        base = ""
        if original_content and original_author:
            base = f"A cité la publication de {original_author} \"{original_content}\""
        elif original_content:
            base = f"A cité une publication \"{original_content}\""
        elif original_author:
            base = f"A cité une publication de {original_author}"
        else:
            base = "A cité une publication"
        if quote_content:
            base += f", avec le commentaire : \"{quote_content}\""
        return base

    def _describe_follow(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f"A suivi l'utilisateur \"{target_user_name}\""
        return "A suivi un utilisateur"

    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if content:
            if post_content and post_author:
                return f"A commenté la publication de {post_author} \"{post_content}\" : \"{content}\""
            elif post_content:
                return f"A commenté la publication \"{post_content}\" : \"{content}\""
            elif post_author:
                return f"A commenté la publication de {post_author} : \"{content}\""
            return f"A commenté : \"{content}\""
        return "A laissé un commentaire"

    def _describe_like_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"A aimé le commentaire de {comment_author} : \"{comment_content}\""
        elif comment_content:
            return f"A aimé un commentaire : \"{comment_content}\""
        elif comment_author:
            return f"A aimé un commentaire de {comment_author}"
        return "A aimé un commentaire"

    def _describe_dislike_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"N'a pas aimé le commentaire de {comment_author} : \"{comment_content}\""
        elif comment_content:
            return f"N'a pas aimé un commentaire : \"{comment_content}\""
        elif comment_author:
            return f"N'a pas aimé un commentaire de {comment_author}"
        return "N'a pas aimé un commentaire"

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"A recherché \"{query}\"" if query else "A effectué une recherche"

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"A recherché l'utilisateur \"{query}\"" if query else "A recherché un utilisateur"

    def _describe_mute(self) -> str:
        target_user_name = self.action_args.get("target_user_name", "")
        if target_user_name:
            return f"A mis en sourdine l'utilisateur \"{target_user_name}\""
        return "A mis en sourdine un utilisateur"

    def _describe_generic(self) -> str:
        return f"A exécuté l'action {self.action_type}"


class GraphMemoryUpdater:
    """
    Service de mise à jour de la mémoire du graphe (via GraphStorage / Neo4j)

    Surveille les journaux d'actions de simulation et envoie les activités des agents au graphe en temps réel.
    Regroupe les activités par plateforme, accumulant BATCH_SIZE activités avant d'envoyer chaque lot.
    """

    BATCH_SIZE = 5

    PLATFORM_DISPLAY_NAMES = {
        'twitter': 'worldinterface1',
        'reddit': 'worldinterface2',
    }

    SEND_INTERVAL = 0.5
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, graph_id: str, storage: GraphStorage):
        """
        Initialiser le mise à jour

        Args:
            graph_id: ID du graphe
            storage: Instance GraphStorage (injectée)
        """
        self.graph_id = graph_id
        self.storage = storage

        self._activity_queue: Queue = Queue()

        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0

        logger.info(f"GraphMemoryUpdater initialisé : graph_id={graph_id}, batch_size={self.BATCH_SIZE}")

    def _get_platform_display_name(self, platform: str) -> str:
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        """Démarrer le thread worker en arrière-plan"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GraphMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"GraphMemoryUpdater démarré : graph_id={self.graph_id}")

    def stop(self):
        """Arrêter le thread worker en arrière-plan"""
        self._running = False

        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(f"GraphMemoryUpdater arrêté : graph_id={self.graph_id}, "
                     f"total_activities={self._total_activities}, "
                     f"batches_sent={self._total_sent}, "
                     f"items_sent={self._total_items_sent}, "
                     f"failed={self._failed_count}, "
                     f"skipped={self._skipped_count}")

    def add_activity(self, activity: AgentActivity):
        """Ajouter une activité d'agent à la file"""
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(f"Activité ajoutée à la file : {activity.agent_name} - {activity.action_type}")

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """Ajouter une activité depuis des données de dictionnaire"""
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        self.add_activity(activity)

    def _worker_loop(self):
        """Boucle worker en arrière-plan - envoyer les activités au graphe par lot par plateforme"""
        while self._running or not self._activity_queue.empty():
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)

                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)

                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]
                            self._send_batch_activities(batch, platform)
                            time.sleep(self.SEND_INTERVAL)

                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Exception dans la boucle worker : {e}")
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Envoyer les activités groupées au graphe en les fusionnant en texte et en utilisant add_text pour déclencher NER.
        """
        if not activities:
            return

        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        for attempt in range(self.MAX_RETRIES):
            try:
                self.storage.add_text(self.graph_id, combined_text)

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"Envoi par lot réussi de {len(activities)} activités {display_name} au graphe {self.graph_id}")
                logger.debug(f"Aperçu du lot : {combined_text[:200]}...")
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"Échec de l'envoi par lot au graphe (tentative {attempt + 1}/{self.MAX_RETRIES}) : {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Échec de l'envoi par lot au graphe après {self.MAX_RETRIES} tentatives : {e}")
                    self._failed_count += 1

    def _flush_remaining(self):
        """Envoyer les activités restantes dans la file et les tampons"""
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display_name = self._get_platform_display_name(platform)
                    logger.info(f"Envoi des {len(buffer)} activités restantes de la plateforme {display_name}")
                    self._send_batch_activities(buffer, platform)
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


class GraphMemoryManager:
    """
    Gère les mises à jour de la mémoire du graphe pour plusieurs simulations.

    Chaque simulation peut avoir sa propre instance de mise à jour indépendante.
    NOTE : create_updater() nécessite une instance GraphStorage — doit être passée en paramètre.
    """

    _updaters: Dict[str, GraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(
        cls, simulation_id: str, graph_id: str, storage: GraphStorage
    ) -> GraphMemoryUpdater:
        """
        Créer un mise à jour de la mémoire du graphe pour une simulation.

        Args:
            simulation_id: ID de la simulation
            graph_id: ID du graphe
            storage: Instance GraphStorage
        """
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = GraphMemoryUpdater(graph_id, storage)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(f"Mise à jour de la mémoire du graphe créée : simulation_id={simulation_id}, graph_id={graph_id}")
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[GraphMemoryUpdater]:
        """Obtenir le mise à jour pour la simulation"""
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Arrêter et supprimer le mise à jour pour la simulation"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"Mise à jour de la mémoire du graphe arrêtée : simulation_id={simulation_id}")

    _stop_all_done = False

    @classmethod
    def stop_all(cls):
        """Arrêter tous les mises à jour"""
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(f"Échec de l'arrêt du mise à jour : simulation_id={simulation_id}, erreur={e}")
                cls._updaters.clear()
            logger.info("Tous les mises à jour de la mémoire du graphe ont été arrêtés")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Obtenir les statistiques de tous les mises à jour"""
        return {
            sim_id: updater.get_stats()
            for sim_id, updater in cls._updaters.items()
        }
