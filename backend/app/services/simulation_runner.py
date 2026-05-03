"""
Exécuteur de simulation OASIS.
Lance les simulations en arrière-plan, enregistre les actions des agents et expose le suivi temps réel.
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from .graph_memory_updater import GraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse

logger = get_logger('mirofish.simulation_runner')

# Indique si la fonction de nettoyage est enregistrée.
_cleanup_registered = False

# Détection de plateforme.
IS_WINDOWS = sys.platform == 'win32'


class RunnerStatus(str, Enum):
    """Statut de l'exécuteur."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Enregistrement d'action d'agent"""
    round_num: int
    timestamp: str
    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Résumé de round"""
    round_num: int
    start_time: str
    end_time: Optional[str] = None
    simulated_hour: int = 0
    twitter_actions: int = 0
    reddit_actions: int = 0
    active_agents: List[int] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """État d'exécution de la simulation (temps réel)"""
    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE

    # Informations de progression
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0

    # Rounds indépendants et temps simulé par plateforme (pour affichage parallèle double plateforme)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # Statut des plateformes
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # Statut d'achèvement des plateformes (détecté via les événements simulation_end dans actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False

    # Résumé des rounds
    rounds: List[RoundSummary] = field(default_factory=list)

    # Actions récentes (pour affichage temps réel sur le frontend)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # Horodatages
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Message d'erreur
    error: Optional[str] = None

    # ID du processus (pour l'arrêt)
    process_pid: Optional[int] = None
    
    def add_action(self, action: AgentAction):
        """Ajouter une action à la liste des actions récentes"""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[:self.max_recent_actions]
        
        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1
        
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(self.current_round / max(self.total_rounds, 1) * 100, 1),
            # Rounds et temps indépendants par plateforme
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        """Détails avec les actions récentes"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Exécuteur de simulation.

    Responsabilités :
    1. Lancer les simulations OASIS dans des processus en arrière-plan.
    2. Analyser les journaux d'exécution et enregistrer les actions de chaque agent.
    3. Fournir les interfaces de suivi en temps réel.
    4. Supporter les opérations pause, arrêt et reprise.
    """
    
    # Répertoire de stockage de l'état d'exécution
    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )
    
    # Répertoire des scripts
    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )
    
    # État d'exécution en mémoire
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # Descripteurs des fichiers stdout
    _stderr_files: Dict[str, Any] = {}  # Descripteurs des fichiers stderr
    
    # Configuration de mise à jour de la mémoire du graphe
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> actif
    
    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Obtenir l'état d'exécution"""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]
        
        # Essayer de charger depuis le fichier
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state
    
    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Charger l'état d'exécution depuis un fichier"""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Rounds et temps indépendants par plateforme
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )

            # Charger les actions récentes
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", ""),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))
            
            return state
        except Exception as e:
            logger.error(f"Échec du chargement de l'état d'exécution : {str(e)}")
            return None
    
    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Sauvegarder l'état d'exécution dans un fichier"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")
        
        data = state.to_detail_dict()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        cls._run_states[state.simulation_id] = state
    
    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # Nombre maximum de rounds de simulation (optionnel, pour tronquer les longues simulations)
        enable_graph_memory_update: bool = False,  # Mettre à jour les activités dans le graphe ou non
        graph_id: str = None,  # ID du graphe (requis lors de l'activation des mises à jour du graphe)
        storage: 'GraphStorage' = None  # Instance GraphStorage (requis si enable_graph_memory_update)
    ) -> SimulationRunState:
        """
        Démarrer la simulation.

        Args:
            simulation_id: ID de la simulation
            platform: Plateforme à exécuter (twitter/reddit/parallel)
            max_rounds: Nombre maximum de rounds de simulation (optionnel, pour tronquer les longues simulations)
            enable_graph_memory_update: Mettre à jour dynamiquement les activités des agents dans le graphe ou non
            graph_id: ID du graphe (requis lors de l'activation des mises à jour du graphe)

        Returns:
            SimulationRunState
        """
        # Vérifier si déjà en cours d'exécution
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [RunnerStatus.RUNNING, RunnerStatus.STARTING]:
            raise ValueError(f"La simulation est déjà en cours d'exécution : {simulation_id}")
        
        # Charger la configuration de la simulation
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            raise ValueError(f"La configuration de simulation n'existe pas, appelez d'abord l'endpoint /prepare")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Initialiser l'état d'exécution
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)
        
        # Si max_rounds est spécifié, tronquer
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(f"Rounds tronqués : {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )
        
        cls._save_run_state(state)
        
        # Si la mise à jour de la mémoire du graphe est activée, créer le mise à jour
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("Vous devez fournir graph_id lors de l'activation de la mise à jour de la mémoire du graphe")
            
            try:
                if not storage:
                    raise ValueError("Vous devez fournir storage (GraphStorage) lors de l'activation de la mise à jour de la mémoire du graphe")
                GraphMemoryManager.create_updater(simulation_id, graph_id, storage)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(f"Mise à jour de la mémoire du graphe activée : simulation_id={simulation_id}, graph_id={graph_id}")
            except Exception as e:
                logger.error(f"Échec de la création du mise à jour de la mémoire du graphe : {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False
        
        # Déterminer quel script exécuter (les scripts se trouvent dans le répertoire backend/scripts/)
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True
        
        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)
        
        if not os.path.exists(script_path):
            raise ValueError(f"Le script n'existe pas : {script_path}")
        
        # Créer la file d'actions
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue
        
        # Démarrer le processus de simulation.
        try:
            # Construire la commande d'exécution avec les chemins complets
            # Nouvelle structure de journaux :
            #   twitter/actions.jsonl - journal d'actions Twitter
            #   reddit/actions.jsonl  - journal d'actions Reddit
            #   simulation.log        - Journal du processus principal
            
            cmd = [
                sys.executable,  # Interpréteur Python
                script_path,
                "--config", config_path,  # Utiliser le chemin complet du fichier de configuration
            ]
            
            # Si max_rounds est spécifié, ajouter aux arguments de la ligne de commande
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])
            
            # Créer le fichier de journal principal pour éviter le débordement du tampon stdout/stderr
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')
            
            # Définir les variables d'environnement du sous-processus pour assurer l'encodage UTF-8 sur Windows
            # Cela corrige les bibliothèques tierces (comme OASIS) qui ne spécifient pas l'encodage lors de la lecture de fichiers
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'  # Support Python 3.7+, rendre tous les open() utilisent UTF-8 par défaut
            env['PYTHONIOENCODING'] = 'utf-8'  # S'assurer que stdout/stderr utilisent UTF-8
            
            # Définir le répertoire de travail sur le répertoire de simulation (les fichiers de base de données etc. seront générés ici)
            # Utiliser start_new_session=True pour créer un nouveau groupe de processus, garantissant que tous les processus enfants peuvent être terminés via os.killpg
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr également écrit dans le même fichier
                text=True,
                encoding='utf-8',  # Spécifier explicitement l'encodage
                bufsize=1,
                env=env,  # Passer les variables d'environnement avec les paramètres UTF-8
                start_new_session=True,  # Créer un nouveau groupe de processus, s'assurer que tous les processus liés se terminent à la fermeture du serveur
            )
            
            # Sauvegarder le descripteur de fichier pour fermeture ultérieure
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # Plus besoin de stderr séparé
            
            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)
            
            # Démarrer le thread de surveillance
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id,),
                daemon=True
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread
            
            logger.info(f"Simulation démarrée avec succès : {simulation_id}, pid={process.pid}, platform={platform}")
            
        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise
        
        return state
    
    @classmethod
    def _monitor_simulation(cls, simulation_id: str):
        """Surveiller le processus de simulation et analyser les journaux d'actions"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        # Nouvelle structure de journaux : journaux d'actions par plateforme
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)
        
        if not process or not state:
            return
        
        twitter_position = 0
        reddit_position = 0
        
        try:
            while process.poll() is None:  # Processus encore en cours
                # Lire le journal d'actions Twitter.
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )
                
                # Lire le journal d'actions Reddit.
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )
                
                # Mettre à jour le statut
                cls._save_run_state(state)
                time.sleep(2)
            
            # Après la fin du processus, lire les journaux une dernière fois
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(twitter_actions_log, twitter_position, state, "twitter")
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(reddit_actions_log, reddit_position, state, "reddit")
            
            # Processus terminé
            exit_code = process.returncode
            
            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
                logger.info(f"Simulation terminée : {simulation_id}")
            else:
                state.runner_status = RunnerStatus.FAILED
                # Lire les informations d'erreur depuis le fichier de journal principal
                main_log_path = os.path.join(sim_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, 'r', encoding='utf-8') as f:
                            error_info = f.read()[-2000:]  # Prendre les 2000 derniers caractères
                except Exception:
                    pass
                state.error = f"Code de sortie du processus : {exit_code}, erreur : {error_info}"
                logger.error(f"Échec de la simulation : {simulation_id}, erreur={state.error}")
            
            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)
            
        except Exception as e:
            logger.error(f"Exception du thread de surveillance : {simulation_id}, erreur={str(e)}")
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
        
        finally:
            # Arrêter le mise à jour de la mémoire du graphe
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    GraphMemoryManager.stop_updater(simulation_id)
                    logger.info(f"Mise à jour de la mémoire du graphe arrêtée : simulation_id={simulation_id}")
                except Exception as e:
                    logger.error(f"Échec de l'arrêt du mise à jour de la mémoire du graphe : {e}")
                cls._graph_memory_enabled.pop(simulation_id, None)
            
            # Nettoyer les ressources du processus
            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)
            
            # Fermer le descripteur de fichier de journal
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)
    
    @classmethod
    def _read_action_log(
        cls, 
        log_path: str, 
        position: int, 
        state: SimulationRunState,
        platform: str
    ) -> int:
        """
        Lire le fichier de journal d'actions
        
        Args:
            log_path: Chemin du fichier de journal d'actions
            position: Dernière position de lecture
            state: Objet d'état d'exécution
            platform: Nom de la plateforme (twitter/reddit)
            
        Returns:
            Nouvelle position de lecture
        """
        # Vérifier si la mise à jour de la mémoire du graphe est activée
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = GraphMemoryManager.get_updater(state.simulation_id)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)
                            
                            # Traiter les entrées de type événement
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")
                                
                                # Détecter l'événement simulation_end, marquer la plateforme comme terminée
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(f"Simulation Twitter terminée : {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(f"Simulation Reddit terminée : {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}")
                                    
                                    # Vérifier si toutes les plateformes activées sont terminées
                                    # Si une seule plateforme est en cours, vérifier uniquement cette plateforme
                                    # Si les deux plateformes sont en cours, les deux doivent être terminées
                                    all_completed = cls._check_all_platforms_completed(state)
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        logger.info(f"Toutes les simulations de plateforme sont terminées : {state.simulation_id}")
                                
                                # Mettre à jour les informations de round (depuis l'événement round_end)
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get("simulated_hours", 0)
                                    
                                    # Mettre à jour les rounds et le temps indépendants par plateforme
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours
                                    
                                    # Les rounds globaux prennent le maximum des deux plateformes
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Le temps global prend le maximum des deux plateformes
                                    state.simulated_hours = max(state.twitter_simulated_hours, state.reddit_simulated_hours)
                                
                                continue
                            
                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)
                            
                            # Mettre à jour les rounds
                            if action.round_num and action.round_num > state.current_round:
                                state.current_round = action.round_num
                            
                            # Si la mise à jour de la mémoire du graphe est activée, envoyer l'activité au graphe
                            if graph_updater:
                                graph_updater.add_activity_from_dict(action_data, platform)
                            
                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Échec de la lecture du journal d'actions : {log_path}, erreur={e}")
            return position
    
    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Vérifier si toutes les plateformes activées ont terminé la simulation
        
        Détermine si une plateforme est activée en vérifiant si le fichier actions.jsonl correspondant existe
        
        Returns:
            True si toutes les plateformes activées sont terminées
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        
        # Vérifier quelles plateformes sont activées (par l'existence des fichiers)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)
        
        # Si une plateforme est activée mais non terminée, retourner False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False
        
        # Au moins une plateforme est activée et terminée
        return twitter_enabled or reddit_enabled
    
    @classmethod
    def _terminate_process(cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10):
        """
        Terminer le processus et ses processus enfants de manière multiplateforme
        
        Args:
            process: Processus à terminer
            simulation_id: ID de la simulation (pour les journaux)
            timeout: Délai d'attente pour la sortie du processus (secondes)
        """
        if IS_WINDOWS:
            # Windows : Utiliser la commande taskkill pour terminer l'arbre de processus
            # /F = terminaison forcée, /T = terminer l'arbre de processus (y compris les processus enfants)
            logger.info(f"Terminer l'arbre de processus (Windows) : simulation={simulation_id}, pid={process.pid}")
            try:
                # Essayer d'abord une terminaison gracieuse
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True,
                    timeout=5
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Terminaison forcée
                    logger.warning(f"Le processus ne répond pas, terminaison forcée : {simulation_id}")
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True,
                        timeout=5
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Échec de taskkill, tentative de terminate : {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix : Utiliser la terminaison de groupe de processus
            # Puisque start_new_session=True, l'ID du groupe de processus est égal au PID du processus principal
            pgid = os.getpgid(process.pid)
            logger.info(f"Terminer le groupe de processus (Unix) : simulation={simulation_id}, pgid={pgid}")
            
            # D'abord envoyer SIGTERM à tout le groupe de processus
            os.killpg(pgid, signal.SIGTERM)
            
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Si toujours pas terminé après le délai, envoyer SIGKILL forcé
                logger.warning(f"Le groupe de processus ne répond pas à SIGTERM, terminaison forcée : {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)
    
    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Arrêter la simulation"""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")
        
        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(f"La simulation n'est pas en cours d'exécution : {simulation_id}, statut={state.runner_status}")
        
        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)
        
        # Terminer le processus
        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                # Le processus n'existe plus
                pass
            except Exception as e:
                logger.error(f"Échec de la terminaison du groupe de processus : {simulation_id}, erreur={e}")
                # Revenir à la terminaison directe du processus
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
        
        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)
        
        # Arrêter le mise à jour de la mémoire du graphe
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                GraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Mise à jour de la mémoire du graphe arrêtée : simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Échec de l'arrêt du mise à jour de la mémoire du graphe : {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)
        
        logger.info(f"Simulation arrêtée : {simulation_id}")
        return state
    
    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Lire les actions depuis un seul fichier d'actions
        
        Args:
            file_path: Chemin du fichier de journal d'actions
            default_platform: Plateforme par défaut (utilisée quand l'enregistrement d'action manque le champ plateforme)
            platform_filter: Filtrer par plateforme
            agent_id: Filtrer par ID d'agent
            round_num: Filtrer par round
        """
        if not os.path.exists(file_path):
            return []
        
        actions = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Ignorer les enregistrements non-action (comme les événements simulation_start, round_start, round_end)
                    if "event_type" in data:
                        continue
                    
                    # Ignorer les enregistrements sans agent_id (actions non-agent)
                    if "agent_id" not in data:
                        continue
                    
                    # Obtenir la plateforme : préférer la plateforme dans l'enregistrement, sinon utiliser la plateforme par défaut
                    record_platform = data.get("platform") or default_platform or ""
                    
                    # Filtrer
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue
                    
                    actions.append(AgentAction(
                        round_num=data.get("round", 0),
                        timestamp=data.get("timestamp", ""),
                        platform=record_platform,
                        agent_id=data.get("agent_id", 0),
                        agent_name=data.get("agent_name", ""),
                        action_type=data.get("action_type", ""),
                        action_args=data.get("action_args", {}),
                        result=data.get("result"),
                        success=data.get("success", True),
                    ))
                    
                except json.JSONDecodeError:
                    continue
        
        return actions
    
    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Obtenir l'historique complet des actions pour toutes les plateformes (sans limite de pagination)
        
        Args:
            simulation_id: ID de la simulation
            platform: Filtrer par plateforme (twitter/reddit)
            agent_id: Filtrer par agent
            round_num: Filtrer par round
            
        Returns:
            Liste complète des actions (triées par horodatage, les plus récentes en premier)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        actions = []
        
        # Lire le fichier d'actions Twitter et définir automatiquement la plateforme.
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(cls._read_actions_from_file(
                twitter_actions_log,
                default_platform="twitter",  # Remplir automatiquement le champ plateforme
                platform_filter=platform,
                agent_id=agent_id, 
                round_num=round_num
            ))
        
        # Lire le fichier d'actions Reddit et définir automatiquement la plateforme.
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(cls._read_actions_from_file(
                reddit_actions_log,
                default_platform="reddit",  # Remplir automatiquement le champ plateforme
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            ))
        
        # Si les fichiers par plateforme n'existent pas, essayer de lire l'ancien format de fichier unique
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,  # Les fichiers d'ancien format devraient avoir le champ plateforme
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num
            )
        
        # Trier par horodatage (les plus récentes en premier)
        actions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return actions
    
    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None
    ) -> List[AgentAction]:
        """
        Obtenir l'historique des actions (avec pagination)
        
        Args:
            simulation_id: ID de la simulation
            limit: Limite du nombre de résultats
            offset: Décalage
            platform: Filtrer par plateforme
            agent_id: Filtrer par agent
            round_num: Filtrer par round
            
        Returns:
            Liste d'actions
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        # Pagination
        return actions[offset:offset + limit]
    
    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round: int = 0,
        end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtenir la chronologie de la simulation (résumée par rounds)
        
        Args:
            simulation_id: ID de la simulation
            start_round: Round de début
            end_round: Round de fin
            
        Returns:
            Informations récapitulatives pour chaque round
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        # Grouper par round
        rounds: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            round_num = action.round_num
            
            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue
            
            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            r = rounds[round_num]
            
            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1
            
            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = r["action_types"].get(action.action_type, 0) + 1
            r["last_action_time"] = action.timestamp
        
        # Convertir en liste
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append({
                "round_num": round_num,
                "twitter_actions": r["twitter_actions"],
                "reddit_actions": r["reddit_actions"],
                "total_actions": r["twitter_actions"] + r["reddit_actions"],
                "active_agents_count": len(r["active_agents"]),
                "active_agents": list(r["active_agents"]),
                "action_types": r["action_types"],
                "first_action_time": r["first_action_time"],
                "last_action_time": r["last_action_time"],
            })
        
        return result
    
    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """
        Obtenir les statistiques pour chaque agent
        
        Returns:
            Liste des statistiques des agents
        """
        actions = cls.get_actions(simulation_id, limit=10000)
        
        agent_stats: Dict[int, Dict[str, Any]] = {}
        
        for action in actions:
            agent_id = action.agent_id
            
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }
            
            stats = agent_stats[agent_id]
            stats["total_actions"] += 1
            
            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1
            
            stats["action_types"][action.action_type] = stats["action_types"].get(action.action_type, 0) + 1
            stats["last_action_time"] = action.timestamp
        
        # Trier par nombre total d'actions
        result = sorted(agent_stats.values(), key=lambda x: x["total_actions"], reverse=True)
        
        return result
    
    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Nettoyer les journaux d'exécution de la simulation (pour redémarrage forcé)
        
        Supprimera les fichiers suivants :
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db (base de données de simulation)
        - reddit_simulation.db (base de données de simulation)
        - env_status.json (statut de l'environnement)
        
        Note : Ne supprime pas les fichiers de configuration (simulation_config.json) ni les fichiers de profils
        
        Args:
            simulation_id: ID de la simulation
            
        Returns:
            Informations sur le résultat du nettoyage
        """
        import shutil
        
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return {"success": True, "message": "Le répertoire de simulation n'existe pas, aucun nettoyage nécessaire"}
        
        cleaned_files = []
        errors = []
        
        # Fichiers à supprimer (y compris les fichiers de base de données)
        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            "twitter_simulation.db",  # Base de données plateforme Twitter
            "reddit_simulation.db",   # Base de données plateforme Reddit
            "env_status.json",        # Fichier de statut de l'environnement
        ]
        
        # Répertoires à supprimer (contient les journaux d'actions)
        dirs_to_clean = ["twitter", "reddit"]
        
        # Supprimer les fichiers
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Échec de la suppression de {filename} : {str(e)}")
        
        # Nettoyer les journaux d'actions dans les répertoires de plateforme
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(f"Échec de la suppression de {dir_name}/actions.jsonl : {str(e)}")
        
        # Nettoyer l'état d'exécution en mémoire
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]
        
        logger.info(f"Nettoyage des journaux de simulation terminé : {simulation_id}, fichiers supprimés : {cleaned_files}")
        
        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None
        }
    
    # Drapeau pour éviter un nettoyage en double
    _cleanup_done = False
    
    @classmethod
    def cleanup_all_simulations(cls):
        """
        Nettoyer tous les processus de simulation en cours
        
        Appelé à la fermeture du serveur, garantit que tous les processus enfants sont terminés
        """
        # Éviter un nettoyage en double
        if cls._cleanup_done:
            return
        cls._cleanup_done = True
        
        # Vérifier s'il y a du contenu à nettoyer (éviter d'imprimer des journaux inutiles pour des processus vides)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)
        
        if not has_processes and not has_updaters:
            return  # Rien à nettoyer, retourner silencieusement
        
        logger.info("Nettoyage de tous les processus de simulation...")
        
        # D'abord arrêter tous les mises à jour de la mémoire du graphe (stop_all imprime les journaux en interne)
        try:
            GraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Échec de l'arrêt du mise à jour de la mémoire du graphe : {e}")
        cls._graph_memory_enabled.clear()
        
        # Copier le dict pour éviter la modification pendant l'itération
        processes = list(cls._processes.items())
        
        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # Processus encore en cours
                    logger.info(f"Terminer le processus de simulation : {simulation_id}, pid={process.pid}")
                    
                    try:
                        # Utiliser la méthode de terminaison de processus multiplateforme
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Le processus peut ne plus exister, essayer la terminaison directe
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()
                    
                    # Mettre à jour run_state.json
                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Serveur fermé, simulation terminée"
                        cls._save_run_state(state)
                    
                    # Mettre à jour également state.json, définir le statut sur arrêté
                    try:
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Tentative de mise à jour de state.json : {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                            state_data['status'] = 'stopped'
                            state_data['updated_at'] = datetime.now().isoformat()
                            with open(state_file, 'w', encoding='utf-8') as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(f"state.json mis à jour au statut arrêté : {simulation_id}")
                        else:
                            logger.warning(f"state.json n'existe pas : {state_file}")
                    except Exception as state_err:
                        logger.warning(f"Échec de la mise à jour de state.json : {simulation_id}, erreur={state_err}")
                        
            except Exception as e:
                logger.error(f"Échec du nettoyage du processus : {simulation_id}, erreur={e}")
        
        # Nettoyer les descripteurs de fichiers
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()
        
        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()
        
        # Nettoyer l'état en mémoire
        cls._processes.clear()
        cls._action_queues.clear()
        
        logger.info("Nettoyage des processus de simulation terminé")
    
    @classmethod
    def register_cleanup(cls):
        """
        Enregistrer la fonction de nettoyage
        
        Appelée au démarrage de l'application Flask, garantit que tous les processus de simulation sont nettoyés à la fermeture du serveur
        """
        global _cleanup_registered
        
        if _cleanup_registered:
            return
        
        # En mode debug Flask, enregistrer le nettoyage uniquement dans le processus enfant du reloader (le processus exécutant réellement l'application)
        # WERKZEUG_RUN_MAIN=true indique qu'il s'agit d'un processus enfant du reloader
        # Si pas en mode debug, pas de variable d'environnement, il faut aussi enregistrer
        is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('WERKZEUG_RUN_MAIN') is not None
        
        # En mode debug, enregistrer uniquement dans le processus enfant du reloader ; toujours enregistrer en mode non-debug
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Marquer comme enregistré, empêcher le processus enfant d'essayer à nouveau
            return
        
        # Sauvegarder les gestionnaires de signaux originaux
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP n'existe que sur les systèmes Unix (macOS/Linux), pas sur Windows
        original_sighup = None
        has_sighup = hasattr(signal, 'SIGHUP')
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)
        
        def cleanup_handler(signum=None, frame=None):
            """Gestionnaire de signal : nettoyer d'abord les processus de simulation, puis appeler le gestionnaire original"""
            # Imprimer les journaux uniquement s'il y a des processus à nettoyer
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Signal {signum} reçu, début du nettoyage...")
            cls.cleanup_all_simulations()
            
            # Appeler le gestionnaire de signal original, laisser Flask se fermer normalement
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP : envoyé à la fermeture du terminal
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Comportement par défaut : quitter normalement
                    sys.exit(0)
            else:
                # Si le gestionnaire original n'est pas appelable (comme SIG_DFL), utiliser le comportement par défaut
                raise KeyboardInterrupt
        
        # Enregistrer le gestionnaire atexit (comme solution de secours)
        atexit.register(cls.cleanup_all_simulations)
        
        # Enregistrer les gestionnaires de signaux (uniquement dans le thread principal)
        try:
            # SIGTERM : signal par défaut pour la commande kill
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT : Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP : fermeture du terminal (Unix uniquement)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # Pas dans le thread principal, ne peut utiliser que atexit
            logger.warning("Impossible d'enregistrer le gestionnaire de signal (pas dans le thread principal), utilisation d'atexit uniquement")
        
        _cleanup_registered = True
    
    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        Obtenir la liste de tous les IDs de simulation en cours d'exécution
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running
    
    # ============== Fonctionnalité d'entretien ==============
    
    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        Vérifier si l'environnement de simulation est actif (peut recevoir des commandes d'entretien)

        Args:
            simulation_id: ID de la simulation

        Returns:
            True signifie que l'environnement est actif, False signifie que l'environnement est fermé
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Obtenir les informations détaillées du statut de l'environnement de simulation

        Args:
            simulation_id: ID de la simulation

        Returns:
            Dictionnaire de détails du statut, contient status, twitter_available, reddit_available, timestamp
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")
        
        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None
        }
        
        if not os.path.exists(status_file):
            return default_status
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp")
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Interroger un seul agent

        Args:
            simulation_id: ID de la simulation
            agent_id: ID de l'agent
            prompt: Question d'entretien
            platform: Spécifier la plateforme (optionnel)
                - "twitter" : interroger uniquement Twitter.
                - "reddit" : interroger uniquement Reddit.
                - None : interroger les deux plateformes en parallèle et retourner des résultats intégrés.
            timeout: Délai d'attente (secondes)

        Returns:
            Dictionnaire de résultat de l'entretien

        Raises:
            ValueError: La simulation n'existe pas ou l'environnement n'est pas en cours d'exécution
            TimeoutError: Délai d'attente dépassé pour la réponse
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"L'environnement de simulation n'est pas en cours d'exécution ou est fermé, impossible d'exécuter l'entretien : {simulation_id}")

        logger.info(f"Envoi de la commande d'entretien : simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}")

        response = ipc_client.send_interview(
            agent_id=agent_id,
            prompt=prompt,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> Dict[str, Any]:
        """
        Interroger plusieurs agents par lot.

        Args:
            simulation_id: ID de la simulation
            interviews: liste d'entretiens, chaque élément contient {"agent_id": int, "prompt": str, "platform": str(optionnel)}.
            platform: plateforme par défaut, surchargée par chaque entrée si renseignée.
                - "twitter" : interroger uniquement Twitter par défaut.
                - "reddit" : interroger uniquement Reddit par défaut.
                - None : interroger chaque agent sur les deux plateformes.
            timeout: Délai d'attente (secondes)

        Returns:
            Dictionnaire de résultat des entretiens par lot.

        Raises:
            ValueError: La simulation n'existe pas ou l'environnement n'est pas en cours d'exécution
            TimeoutError: Délai d'attente dépassé pour la réponse
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(f"L'environnement de simulation n'est pas en cours d'exécution ou est fermé, impossible d'exécuter l'entretien : {simulation_id}")

        logger.info(f"Envoi de la commande d'entretien par lot : simulation_id={simulation_id}, count={len(interviews)}, platform={platform}")

        response = ipc_client.send_batch_interview(
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp
            }
    
    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: str = None,
        timeout: float = 180.0
    ) -> Dict[str, Any]:
        """
        Interroger tous les agents avec une question globale.

        Interviewer tous les agents de la simulation en utilisant la même question

        Args:
            simulation_id: ID de la simulation
            prompt: Question d'entretien (tous les agents utilisent la même question)
            platform: Spécifier la plateforme (optionnel)
                - "twitter" : interroger uniquement Twitter.
                - "reddit" : interroger uniquement Reddit.
                - None : interroger chaque agent sur les deux plateformes.
            timeout: Délai d'attente (secondes)

        Returns:
            Dictionnaire de résultat de l'entretien global.
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")

        # Obtenir les informations de tous les agents depuis le fichier de configuration
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"La configuration de simulation n'existe pas : {simulation_id}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(f"Aucun agent dans la configuration de simulation : {simulation_id}")

        # Construire la liste d'entretiens par lot.
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({
                    "agent_id": agent_id,
                    "prompt": prompt
                })

        logger.info(f"Envoi de la commande d'entretien global : simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}")

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout
        )
    
    @classmethod
    def close_simulation_env(
        cls,
        simulation_id: str,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Fermer l'environnement de simulation sans arrêter brutalement le processus.
        
        Envoyer une commande de fermeture d'environnement à la simulation pour quitter gracieusement le mode d'attente de commandes
        
        Args:
            simulation_id: ID de la simulation
            timeout: Délai d'attente (secondes)
            
        Returns:
            Dictionnaire de résultat de l'opération
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")
        
        ipc_client = SimulationIPCClient(sim_dir)
        
        if not ipc_client.check_env_alive():
            return {
                "success": True,
                "message": "Environnement déjà fermé"
            }
        
        logger.info(f"Envoi de la commande de fermeture d'environnement : simulation_id={simulation_id}")
        
        try:
            response = ipc_client.send_close_env(timeout=timeout)
            
            return {
                "success": response.status.value == "completed",
                "message": "Commande de fermeture d'environnement envoyée",
                "result": response.result,
                "timestamp": response.timestamp
            }
        except TimeoutError:
            # Le délai peut être dû à la fermeture de l'environnement
            return {
                "success": True,
                "message": "Commande de fermeture d'environnement envoyée (délai d'attente dépassé pour la réponse, l'environnement est peut-être en cours de fermeture)"
            }
    
    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtenir l'historique des entretiens depuis une seule base de données"""
        import sqlite3
        
        if not os.path.exists(db_path):
            return []
        
        results = []
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if agent_id is not None:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (agent_id, limit))
            else:
                cursor.execute("""
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
            
            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}
                
                results.append({
                    "agent_id": user_id,
                    "response": info.get("response", info),
                    "prompt": info.get("prompt", ""),
                    "timestamp": created_at,
                    "platform": platform_name
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Échec de la lecture de l'historique des entretiens ({platform_name}) : {e}")
        
        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: str = None,
        agent_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Obtenir l'historique des entretiens (lu depuis la base de données)
        
        Args:
            simulation_id: ID de la simulation
            platform: Type de plateforme (reddit/twitter/None)
                - "reddit" : obtenir uniquement l'historique de la plateforme Reddit
                - "twitter" : obtenir uniquement l'historique de la plateforme Twitter
                - None : obtenir tout l'historique des deux plateformes
            agent_id: Spécifier l'ID de l'agent (optionnel, obtenir uniquement l'historique de cet agent)
            limit: Limite du nombre de résultats par plateforme
            
        Returns:
            Liste des enregistrements d'historique des entretiens
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        
        results = []
        
        # Déterminer les plateformes à interroger
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # Quand la plateforme n'est pas spécifiée, interroger les deux plateformes
            platforms = ["twitter", "reddit"]
        
        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path,
                platform_name=p,
                agent_id=agent_id,
                limit=limit
            )
            results.extend(platform_results)
        
        # Trier par temps en ordre décroissant
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Si plusieurs plateformes ont été interrogées, limiter le nombre total
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]
        
        return results
