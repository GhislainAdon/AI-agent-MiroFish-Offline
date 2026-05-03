"""
Script de simulation Twitter OASIS
Ce script lit les paramètres du fichier de configuration pour exécuter la simulation, atteignant une automatisation complète

Fonctionnalités :
- Maintenir l'environnement en cours d'exécution après la fin de la simulation, entrer en mode d'attente
- Support des commandes Interview via IPC
- Support de l'interview d'un Agent unique et de l'interview par lot
- Support de la commande d'arrêt à distance de l'environnement

Utilisation :
    python run_twitter_simulation.py --config /chemin/vers/simulation_config.json
    python run_twitter_simulation.py --config /chemin/vers/simulation_config.json --no-wait  # Fermer immédiatement après la fin
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# Variables globales : pour la gestion des signaux
_shutdown_event = None
_cleanup_done = False

# Ajouter les chemins du projet
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Charger le fichier .env depuis la racine du projet (contient LLM_API_KEY et autres configurations)
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """Formateur personnalisé pour convertir les séquences d'échappement Unicode en caractères lisibles"""
    
    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
    
    def format(self, record):
        result = super().format(record)
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """Filtrer les avertissements max_tokens de camel-ai (nous ne définissons pas intentionnellement max_tokens pour laisser le modèle décider)"""
    
    def filter(self, record):
        # Filtrer les journaux contenant des avertissements max_tokens
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# Ajouter le filtre immédiatement au chargement du module, pour qu'il prenne effet avant l'exécution du code camel
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """Configurer la journalisation OASIS, utiliser des fichiers de journal avec noms fixes"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Nettoyer les anciens fichiers de journal
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    
    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }
    
    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph
    )
except ImportError as e:
    print(f"Erreur : Dépendance manquante {e}")
    print("Veuillez installer d'abord : pip install oasis-ai camel-ai")
    sys.exit(1)


# Constantes liées à l'IPC
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """Constantes de type de commande"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class IPCHandler:
    """Gestionnaire de commandes IPC"""
    
    def __init__(self, simulation_dir: str, env, agent_graph):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True
        
        # S'assurer que les répertoires existent
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """Mettre à jour le statut de l'environnement"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Interroger les commandes en attente"""
        if not os.path.exists(self.commands_dir):
            return None
        
        # Obtenir les fichiers de commande (triés par heure)
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Envoyer une réponse"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # Supprimer le fichier de commande
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str) -> bool:
        """
        Gérer la commande d'interview d'un Agent unique
        
        Retourne :
            True signifie succès, False signifie échec
        """
        try:
            # Obtenir l'Agent
            agent = self.agent_graph.get_agent(agent_id)
            
            # Créer l'action Interview
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            
            # Exécuter l'Interview
            actions = {agent: interview_action}
            await self.env.step(actions)
            
            # Obtenir le résultat depuis la base de données
            result = self._get_interview_result(agent_id)
            
            self.send_response(command_id, "completed", result=result)
            print(f"  Interview terminée : agent_id={agent_id}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Interview échouée : agent_id={agent_id}, erreur={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict]) -> bool:
        """
        Gérer la commande d'interview par lot
        
        Args :
            interviews : [{"agent_id": int, "prompt": str}, ...]
        """
        try:
            # Construire le dictionnaire d'actions
            actions = {}
            agent_prompts = {}  # Enregistrer le prompt de chaque agent
            
            for interview in interviews:
                agent_id = interview.get("agent_id")
                prompt = interview.get("prompt", "")
                
                try:
                    agent = self.agent_graph.get_agent(agent_id)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW,
                        action_args={"prompt": prompt}
                    )
                    agent_prompts[agent_id] = prompt
                except Exception as e:
                    print(f"  Avertissement : Impossible d'obtenir l'Agent {agent_id} : {e}")
            
            if not actions:
                self.send_response(command_id, "failed", error="Aucun Agent valide")
                return False
            
            # Exécuter l'Interview par lot
            await self.env.step(actions)
            
            # Obtenir tous les résultats
            results = {}
            for agent_id in agent_prompts.keys():
                result = self._get_interview_result(agent_id)
                results[agent_id] = result
            
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Interview par lot terminée : {len(results)} Agents")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Interview par lot échouée : {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """Obtenir le dernier résultat d'Interview depuis la base de données"""
        db_path = os.path.join(self.simulation_dir, "twitter_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
            return result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Interroger le dernier enregistrement Interview
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))
            
            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            
            conn.close()
            
        except Exception as e:
            print(f"  Échec de la lecture du résultat Interview : {e}")
        
        return result
    
    async def process_commands(self) -> bool:
        """
        Traiter toutes les commandes en attente
        
        Retourne :
            True signifie continuer l'exécution, False signifie qu'il faut quitter
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\nCommande IPC reçue : {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", "")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", [])
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Commande de fermeture de l'environnement reçue")
            self.send_response(command_id, "completed", result={"message": "L'environnement va se fermer"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"Type de commande inconnu : {command_type}")
            return True


class TwitterSimulationRunner:
    """Lanceur de simulation Twitter"""
    
    # Actions Twitter disponibles (INTERVIEW non incluse, INTERVIEW ne peut être déclenchée que manuellement via ManualAction)
    AVAILABLE_ACTIONS = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]
    
    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        Initialiser le lanceur de simulation
        
        Args :
            config_path : Chemin du fichier de configuration (simulation_config.json)
            wait_for_commands : S'il faut attendre des commandes après la fin de la simulation (True par défaut)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Charger le fichier de configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_profile_path(self) -> str:
        """Obtenir le chemin du fichier de profil (Twitter OASIS utilise le format CSV)"""
        return os.path.join(self.simulation_dir, "twitter_profiles.csv")
    
    def _get_db_path(self) -> str:
        """Obtenir le chemin de la base de données"""
        return os.path.join(self.simulation_dir, "twitter_simulation.db")
    
    def _create_model(self):
        """
        Créer le modèle LLM
        
        Utilisation unifiée de la configuration dans le fichier .env à la racine du projet (priorité la plus élevée) :
        - LLM_API_KEY : Clé API
        - LLM_BASE_URL : URL de base de l'API
        - LLM_MODEL_NAME : Nom du modèle
        """
        # Lire la configuration depuis .env en premier
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        
        # Si absent du .env, utiliser la configuration comme solution de secours
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
        # Définir les variables d'environnement requises par camel-ai
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Configuration de clé API manquante, veuillez définir LLM_API_KEY dans le fichier .env à la racine du projet")
        
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        
        print(f"Configuration LLM : modèle={llm_model}, url_base={llm_base_url[:40] if llm_base_url else 'défaut'}...")
        
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )
    
    def _get_active_agents_for_round(
        self, 
        env, 
        current_hour: int,
        round_num: int
    ) -> List:
        """
        Décider quels Agents activer ce tour en fonction de l'heure et de la configuration
        
        Args :
            env : Environnement OASIS
            current_hour : Heure de simulation actuelle (0-23)
            round_num : Numéro du tour actuel
            
        Retourne :
            Liste des Agents activés
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        # Nombre de base d'activations
        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        
        # Ajuster selon la période
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        
        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        
        target_count = int(random.uniform(base_min, base_max) * multiplier)
        
        # Calculer la probabilité d'activation selon la configuration de chaque Agent
        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            
            # Vérifier si dans les heures d'activité
            if current_hour not in active_hours:
                continue
            
            # Calculer la probabilité selon le niveau d'activité
            if random.random() < activity_level:
                candidates.append(agent_id)
        
        # Sélection aléatoire
        selected_ids = random.sample(
            candidates, 
            min(target_count, len(candidates))
        ) if candidates else []
        
        # Convertir en objets Agent
        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass
        
        return active_agents
    
    async def run(self, max_rounds: int = None):
        """Exécuter la simulation Twitter
        
        Args :
            max_rounds : Nombre maximum de tours de simulation (optionnel, utilisé pour tronquer les longues simulations)
        """
        print("=" * 60)
        print("Simulation Twitter OASIS")
        print(f"Fichier de configuration : {self.config_path}")
        print(f"ID de simulation : {self.config.get('simulation_id', 'inconnu')}")
        print(f"Mode d'attente : {'Activé' if self.wait_for_commands else 'Désactivé'}")
        print("=" * 60)
        
        # Charger la configuration temporelle
        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        
        # Calculer le nombre total de tours
        total_rounds = (total_hours * 60) // minutes_per_round
        
        # Si le nombre maximum de tours est spécifié, tronquer
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\nTours tronqués : {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        print(f"\nParamètres de simulation :")
        print(f"  - Durée totale de simulation : {total_hours} heures")
        print(f"  - Temps par tour : {minutes_per_round} minutes")
        print(f"  - Total des tours : {total_rounds}")
        if max_rounds:
            print(f"  - Limite maximale de tours : {max_rounds}")
        print(f"  - Nombre d'Agents : {len(self.config.get('agent_configs', []))}")
        
        # Créer le modèle
        print("\nInitialisation du modèle LLM...")
        model = self._create_model()
        
        # Charger le graphe des Agents
        print("Chargement du profil des Agents...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"Erreur : Le fichier de profil n'existe pas : {profile_path}")
            return
        
        self.agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )
        
        # Chemin de la base de données
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Ancienne base de données supprimée : {db_path}")
        
        # Créer l'environnement
        print("Création de l'environnement OASIS...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            semaphore=30,  # Limiter le nombre maximal de requêtes LLM simultanées pour éviter la surcharge de l'API
        )
        
        await self.env.reset()
        print("Initialisation de l'environnement terminée\n")
        
        # Initialiser le gestionnaire IPC
        self.ipc_handler = IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")
        
        # Exécuter les événements initiaux
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        
        if initial_posts:
            print(f"Exécution des événements initiaux ({len(initial_posts)} publications initiales)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                except Exception as e:
                    print(f"  Avertissement : Impossible de créer la publication initiale pour l'Agent {agent_id} : {e}")
            
            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  {len(initial_actions)} publications initiales publiées")
        
        # Boucle de simulation principale
        print("\nDémarrage de la boucle de simulation...")
        start_time = datetime.now()
        
        for round_num in range(total_rounds):
            # Calculer l'heure de simulation actuelle
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1
            
            # Obtenir les Agents activés ce tour
            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )
            
            if not active_agents:
                continue
            
            # Construire les actions
            actions = {
                agent: LLMAction()
                for _, agent in active_agents
            }
            
            # Exécuter les actions
            await self.env.step(actions)
            
            # Afficher la progression
            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(f"  [Jour {simulated_day}, {simulated_hour:02d}:00] "
                      f"Tour {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                      f"- {len(active_agents)} agents actifs "
                      f"- temps écoulé : {elapsed:.1f}s")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nBoucle de simulation terminée !")
        print(f"  - Temps total : {total_elapsed:.1f} secondes")
        print(f"  - Base de données : {db_path}")
        
        # S'il faut entrer en mode d'attente
        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("Entrée en mode d'attente - l'environnement reste en cours d'exécution")
            print("Commandes supportées : interview, batch_interview, close_env")
            print("=" * 60)
            
            self.ipc_handler.update_status("alive")
            
            # Boucle d'attente de commandes (utilisant le _shutdown_event global)
            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break  # Signal de sortie reçu
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\nSignal d'interruption reçu")
            except asyncio.CancelledError:
                print("\nTâche annulée")
            except Exception as e:
                print(f"\nErreur lors du traitement de la commande : {e}")
            
            print("\nFermeture de l'environnement...")
        
        # Fermer l'environnement
        self.ipc_handler.update_status("stopped")
        await self.env.close()
        
        print("Environnement fermé")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='Simulation Twitter OASIS')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Chemin du fichier de configuration (simulation_config.json)'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Nombre maximum de tours de simulation (optionnel, utilisé pour tronquer les longues simulations)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Fermer l\'environnement immédiatement après la fin de la simulation, ne pas entrer en mode d\'attente'
    )
    
    args = parser.parse_args()
    
    # Créer l'événement d'arrêt au début de la fonction main
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Erreur : Le fichier de configuration n'existe pas : {args.config}")
        sys.exit(1)
    
    # Initialiser la configuration de journalisation (utiliser des noms de fichiers fixes, nettoyer les anciens journaux)
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))
    
    runner = TwitterSimulationRunner(
        config_path=args.config,
        wait_for_commands=not args.no_wait
    )
    await runner.run(max_rounds=args.max_rounds)


def setup_signal_handlers():
    """
    Configurer les gestionnaires de signaux pour assurer une sortie propre lors de la réception de SIGTERM/SIGINT
    Donner au programme une chance de nettoyer correctement les ressources (fermer la base de données, l'environnement, etc.)
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nSignal {sig_name} reçu, fermeture en cours...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            # Forcer la sortie uniquement après avoir reçu le signal de manière répétée
            print("Forcer la sortie...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgramme interrompu")
    except SystemExit:
        pass
    finally:
        print("Processus de simulation terminé")
