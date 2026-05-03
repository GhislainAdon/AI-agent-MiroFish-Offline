"""
Module de communication inter-processus de simulation pour la communication entre Flask et le script de simulation.

La communication utilise un modèle simple de commande/réponse basé sur le système de fichiers :
1. Flask écrit les commandes dans le répertoire commands/.
2. Le script de simulation scrute le répertoire des commandes, exécute la commande, et écrit la réponse dans le répertoire responses/.
3. Flask scrute le répertoire des réponses et récupère le résultat.
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger('mirofish.simulation_ipc')


class CommandType(str, Enum):
    """Type de commande"""
    INTERVIEW = "interview"           # Entretien d'un seul agent
    BATCH_INTERVIEW = "batch_interview"  # Entretien par lot
    CLOSE_ENV = "close_env"           # Fermer l'environnement


class CommandStatus(str, Enum):
    """Statut de la commande"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IPCCommand:
    """Commande IPC"""
    command_id: str
    command_type: CommandType
    args: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type.value,
            "args": self.args,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCCommand':
        return cls(
            command_id=data["command_id"],
            command_type=CommandType(data["command_type"]),
            args=data.get("args", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class IPCResponse:
    """Réponse IPC"""
    command_id: str
    status: CommandStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCResponse':
        return cls(
            command_id=data["command_id"],
            status=CommandStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


class SimulationIPCClient:
    """
    Client IPC de simulation côté Flask.

    Utilisé pour envoyer des commandes au processus de simulation et attendre les réponses.
    """
    
    def __init__(self, simulation_dir: str):
        """
        Initialiser le client IPC
        
        Args:
            simulation_dir: Répertoire de données de simulation
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # S'assurer que les répertoires existent
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def send_command(
        self,
        command_type: CommandType,
        args: Dict[str, Any],
        timeout: float = 60.0,
        poll_interval: float = 0.5
    ) -> IPCResponse:
        """
        Envoyer une commande et attendre la réponse
        
        Args:
            command_type: Type de commande
            args: Arguments de la commande
            timeout: Délai d'attente (secondes)
            poll_interval: Intervalle de scrutation (secondes)
            
        Returns:
            IPCResponse
            
        Raises:
            TimeoutError: Délai d'attente dépassé pour la réponse
        """
        command_id = str(uuid.uuid4())
        command = IPCCommand(
            command_id=command_id,
            command_type=command_type,
            args=args
        )
        
        # Écrire le fichier de commande
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Commande IPC envoyée : {command_type.value}, command_id={command_id}")
        
        # Attendre la réponse
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    response = IPCResponse.from_dict(response_data)

                    # Nettoyer les fichiers de commande et de réponse
                    try:
                        os.remove(command_file)
                        os.remove(response_file)
                    except OSError:
                        pass
                    
                    logger.info(f"Réponse IPC reçue : command_id={command_id}, statut={response.status.value}")
                    return response
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Échec de l'analyse de la réponse : {e}")
            
            time.sleep(poll_interval)
        
        # Délai dépassé
        logger.error(f"Délai d'attente dépassé pour la réponse IPC : command_id={command_id}")
        
        # Nettoyer le fichier de commande
        try:
            os.remove(command_file)
        except OSError:
            pass
        
        raise TimeoutError(f"Délai d'attente dépassé pour la réponse à la commande ({timeout} secondes)")
    
    def send_interview(
        self,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> IPCResponse:
        """
        Envoyer une commande d'entretien d'un seul agent
        
        Args:
            agent_id: ID de l'agent
            prompt: Question d'entretien
            platform: Spécifier la plateforme (optionnel)
                - "twitter" : interroger uniquement la plateforme Twitter
                - "reddit" : interroger uniquement la plateforme Reddit  
                - None : interroger les deux plateformes simultanément dans les simulations double plateforme, plateforme unique dans les simulations mono plateforme
            timeout: Délai d'attente
            
        Returns:
            IPCResponse, le champ result contient le résultat de l'entretien
        """
        args = {
            "agent_id": agent_id,
            "prompt": prompt
        }
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_batch_interview(
        self,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> IPCResponse:
        """
        Envoyer une commande d'entretien par lot
        
        Args:
            interviews: Liste d'entretiens, chaque élément contient {"agent_id": int, "prompt": str, "platform": str(optionnel)}
            platform: Plateforme par défaut (optionnel, surchargée par la plateforme de chaque élément d'entretien)
                - "twitter" : interroger uniquement la plateforme Twitter par défaut
                - "reddit" : interroger uniquement la plateforme Reddit par défaut
                - None : interroger chaque agent sur les deux plateformes simultanément dans les simulations double plateforme
            timeout: Délai d'attente
            
        Returns:
            IPCResponse, le champ result contient tous les résultats des entretiens
        """
        args = {"interviews": interviews}
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.BATCH_INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_close_env(self, timeout: float = 30.0) -> IPCResponse:
        """
        Envoyer une commande de fermeture d'environnement
        
        Args:
            timeout: Délai d'attente
            
        Returns:
            IPCResponse
        """
        return self.send_command(
            command_type=CommandType.CLOSE_ENV,
            args={},
            timeout=timeout
        )
    
    def check_env_alive(self) -> bool:
        """
        Vérifier si l'environnement de simulation est actif
        
        Détermine en vérifiant le fichier env_status.json
        """
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        if not os.path.exists(status_file):
            return False
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return status.get("status") == "alive"
        except (json.JSONDecodeError, OSError):
            return False


class SimulationIPCServer:
    """
    Serveur IPC de simulation côté script de simulation.

    Scrute le répertoire des commandes, exécute les commandes, et retourne les réponses.
    """
    
    def __init__(self, simulation_dir: str):
        """
        Initialiser le serveur IPC
        
        Args:
            simulation_dir: Répertoire de données de simulation
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # S'assurer que les répertoires existent
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # Statut de l'environnement
        self._running = False
    
    def start(self):
        """Marquer le serveur comme démarré"""
        self._running = True
        self._update_env_status("alive")
    
    def stop(self):
        """Marquer le serveur comme arrêté"""
        self._running = False
        self._update_env_status("stopped")
    
    def _update_env_status(self, status: str):
        """Mettre à jour le fichier de statut de l'environnement"""
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_commands(self) -> Optional[IPCCommand]:
        """
        Scruter le répertoire des commandes, retourner la première commande en attente
        
        Returns:
            IPCCommand ou None
        """
        if not os.path.exists(self.commands_dir):
            return None
        
        # Obtenir les fichiers de commande triés par date
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return IPCCommand.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Échec de la lecture du fichier de commande : {filepath}, {e}")
                continue
        
        return None
    
    def send_response(self, response: IPCResponse):
        """
        Envoyer une réponse
        
        Args:
            response: Réponse IPC
        """
        response_file = os.path.join(self.responses_dir, f"{response.command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Supprimer le fichier de commande
        command_file = os.path.join(self.commands_dir, f"{response.command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def send_success(self, command_id: str, result: Dict[str, Any]):
        """Envoyer une réponse de succès"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.COMPLETED,
            result=result
        ))
    
    def send_error(self, command_id: str, error: str):
        """Envoyer une réponse d'erreur"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.FAILED,
            error=error
        ))
