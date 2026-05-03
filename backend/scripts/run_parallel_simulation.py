"""
Script de simulation parallèle double plateforme OASIS
Exécuter les simulations Twitter et Reddit simultanément avec le même fichier de configuration

Fonctionnalités :
- Simulation parallèle double plateforme (Twitter + Reddit)
- Maintenir l'environnement en cours d'exécution après la fin de la simulation (entrer en mode d'attente)
- Support des commandes Interview via IPC
- Support de l'interview d'un Agent unique et de l'interview par lot
- Support de la commande d'arrêt à distance de l'environnement

Utilisation :
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # Fermer immédiatement après la fin
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

Structure des journaux :
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Journal des actions de la plateforme Twitter
    ├── reddit/
    │   └── actions.jsonl    # Journal des actions de la plateforme Reddit
    ├── simulation.log       # Journal du processus principal de simulation
    └── run_state.json       # État d'exécution (pour les requêtes API)
"""

# ============================================================
# Correction du problème d'encodage Windows : Définir l'encodage UTF-8 avant toutes les importations
# Ceci corrige le problème où la bibliothèque tierce OASIS ne spécifie pas l'encodage lors de la lecture des fichiers
# ============================================================
import sys
import os

if sys.platform == 'win32':
    # Définir l'encodage E/S par défaut de Python sur UTF-8
    # Ceci affecte tous les appels open() sans encodage spécifié
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

    # Reconfigurer le flux de sortie standard en UTF-8 (corriger les problèmes d'encodage de la console)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Forcer l'encodage par défaut (affecte l'encodage par défaut de la fonction open())
    # Note : Cela doit être défini au démarrage de Python, la configuration à l'exécution peut ne pas fonctionner
    # Nous devons donc aussi monkey-patcher la fonction open intégrée
    import builtins
    _original_open = builtins.open

    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        """
        Envelopper la fonction open() pour utiliser l'encodage UTF-8 par défaut en mode texte
        Ceci peut corriger le problème où les bibliothèques tierces (comme OASIS) ne spécifient pas l'encodage lors de la lecture des fichiers
        """
        # Ne définir l'encodage par défaut que pour le mode texte (non binaire) sans encodage spécifié
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors,
                              newline, closefd, opener)

    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import multiprocessing
import random
import signal
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# Variables globales : pour la gestion des signaux
_shutdown_event = None
_cleanup_done = False

# Ajouter le répertoire backend au chemin
# Le script est fixé dans le répertoire backend/scripts/
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
    print(f"Configuration d'environnement chargée : {_env_file}")
else:
    # Essayer de charger backend/.env
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Configuration d'environnement chargée : {_backend_env}")


class MaxTokensWarningFilter(logging.Filter):
    """Filtrer les avertissements max_tokens de camel-ai (nous ne définissons pas intentionnellement max_tokens pour laisser le modèle décider)"""

    def filter(self, record):
        # Filtrer les journaux contenant des avertissements max_tokens
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# Ajouter le filtre immédiatement au chargement du module, pour qu'il prenne effet avant l'exécution du code camel
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    Désactiver la sortie de journalisation verbeuse de la bibliothèque OASIS
    La journalisation OASIS est trop verbeuse (journalise chaque observation et action de l'agent), nous utilisons notre propre action_logger
    """
    # Désactiver tous les journaux OASIS
    oasis_loggers = [
        "social.agent",
        "social.twitter",
        "social.rec",
        "oasis.env",
        "table",
    ]

    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # Ne journaliser que les erreurs critiques
        logger.handlers.clear()
        logger.propagate = False


def init_logging_for_simulation(simulation_dir: str):
    """
    Initialiser la configuration du journal de simulation

    Args :
        simulation_dir : Chemin du répertoire de simulation
    """
    # Désactiver la journalisation verbeuse OASIS
    disable_oasis_logging()

    # Nettoyer l'ancien répertoire de journaux (s'il existe)
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager, PlatformActionLogger

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"Erreur : Dépendance manquante {e}")
    print("Veuillez installer d'abord : pip install oasis-ai camel-ai")
    sys.exit(1)


# Actions Twitter disponibles (INTERVIEW non incluse, INTERVIEW ne peut être déclenchée que manuellement via ManualAction)
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Actions Reddit disponibles (INTERVIEW non incluse, INTERVIEW ne peut être déclenchée que manuellement via ManualAction)
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
]


# Constantes liées à l'IPC
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """Constantes de type de commande"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """
    Gestionnaire de commandes IPC double plateforme
    
    Gérer les environnements des deux plateformes, gérer les commandes Interview
    """
    
    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None
    ):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        
        # S'assurer que le répertoire existe
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """Mettre à jour le statut de l'environnement"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
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
    
    def _get_env_and_graph(self, platform: str):
        """
        Obtenir l'environnement et l'agent_graph pour la plateforme spécifiée
        
        Args :
            platform : Nom de la plateforme ("twitter" ou "reddit")
            
        Retourne :
            (env, agent_graph, nom_plateforme) ou (None, None, None)
        """
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None
    
    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict[str, Any]:
        """
        Exécuter un Interview sur une seule plateforme
        
        Retourne :
            Dictionnaire contenant le résultat, ou dictionnaire contenant l'erreur
        """
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        
        if not env or not agent_graph:
            return {"platform": platform, "error": f"Plateforme {platform} indisponible"}
        
        try:
            agent = agent_graph.get_agent(agent_id)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            actions = {agent: interview_action}
            await env.step(actions)
            
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
            
        except Exception as e:
            return {"platform": platform, "error": str(e)}
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str, platform: str = None) -> bool:
        """
        Gérer la commande d'interview d'un Agent unique
        
        Args :
            command_id : ID de commande
            agent_id : ID de l'Agent
            prompt : Question d'interview
            platform : Spécifier la plateforme (optionnel)
                - "twitter" : Interview uniquement la plateforme Twitter
                - "reddit" : Interview uniquement la plateforme Reddit
                - None/non spécifié : Interview les deux plateformes simultanément, retourner le résultat intégré
            
        Retourne :
            True signifie succès, False signifie échec
        """
        # Si la plateforme est spécifiée, n'interviewer que cette plateforme
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(f"  Interview échouée : agent_id={agent_id}, plateforme={platform}, erreur={result['error']}")
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(f"  Interview terminée : agent_id={agent_id}, plateforme={platform}")
                return True
        
        # Plateforme non spécifiée : interviewer les deux plateformes simultanément
        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="Aucun environnement de simulation disponible")
            return False
        
        results = {
            "agent_id": agent_id,
            "prompt": prompt,
            "platforms": {}
        }
        success_count = 0
        
        # Interviewer les deux plateformes en parallèle
        tasks = []
        platforms_to_interview = []
        
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")
        
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")
        
        # Exécuter en parallèle
        platform_results = await asyncio.gather(*tasks)
        
        for platform_name, platform_result in zip(platforms_to_interview, platform_results):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1
        
        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(f"  Interview terminée : agent_id={agent_id}, plateformes_réussies={success_count}/{len(platforms_to_interview)}")
            return True
        else:
            errors = [f"{p} : {r.get('error', 'Erreur inconnue')}" for p, r in results["platforms"].items()]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(f"  Interview échouée : agent_id={agent_id}, Toutes les plateformes ont échoué")
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict], platform: str = None) -> bool:
        """
        Gérer la commande d'interview par lot
        
        Args :
            command_id : ID de commande
            interviews : [{"agent_id": int, "prompt": str, "platform": str(optionnel)}, ...]
            platform : plateforme par défaut (peut être remplacée par chaque élément d'interview)
                - "twitter" : Interview uniquement la plateforme Twitter
                - "reddit" : Interview uniquement la plateforme Reddit
                - None/non spécifié : Interview les deux plateformes simultanément pour chaque Agent
        """
        # Grouper par plateforme
        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []  # Besoin d'interviewer les deux plateformes simultanément
        
        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:
                # Plateforme non spécifiée : interviewer les deux plateformes
                both_platforms_interviews.append(interview)
        
        # Répartir both_platforms_interviews sur les deux plateformes
        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)
        
        results = {}
        
        # Gérer l'interview de la plateforme Twitter
        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Avertissement : Impossible d'obtenir l'Agent Twitter {agent_id} : {e}")
                
                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)
                    
                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  Interview par lot Twitter échouée : {e}")
        
        # Gérer l'interview de la plateforme Reddit
        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  Avertissement : Impossible d'obtenir l'Agent Reddit {agent_id} : {e}")
                
                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)
                    
                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  Interview par lot Reddit échouée : {e}")
        
        if results:
            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Interview par lot terminée : {len(results)} Agents")
            return True
        else:
            self.send_response(command_id, "failed", error="Aucune interview réussie")
            return False
    
    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """Obtenir le dernier résultat d'Interview depuis la base de données"""
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")
        
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
                args.get("prompt", ""),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("Commande de fermeture de l'environnement reçue")
            self.send_response(command_id, "completed", result={"message": "L'environnement va se fermer"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"Type de commande inconnu : {command_type}")
            return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Charger le fichier de configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Types d'actions non essentielles à filtrer (ces actions ont une faible valeur analytique)
FILTERED_ACTIONS = {'refresh', 'sign_up'}

# Table de correspondance des types d'actions (Nom de base de données -> nom standard)
ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}


def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Obtenir la correspondance agent_id -> entity_name depuis simulation_config
    
    Ceci permet d'afficher les vrais noms d'entités dans actions.jsonl au lieu de codes comme "Agent_0"
    
    Args :
        config : Contenu de simulation_config.json
        
    Retourne :
        Dictionnaire de correspondance agent_id -> entity_name
    """
    agent_names = {}
    agent_configs = config.get("agent_configs", [])
    
    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name
    
    return agent_names


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Obtenir les nouveaux enregistrements d'actions depuis la base de données et compléter les informations de contexte
    
    Args :
        db_path : Chemin du fichier de base de données
        last_rowid : Valeur maximale de rowid de la dernière lecture (utiliser rowid au lieu de created_at car différentes plateformes ont différents formats de created_at)
        agent_names : Correspondance agent_id -> agent_name
        
    Retourne :
        (actions_list, new_last_rowid)
        - actions_list : Liste d'actions, chaque élément contient agent_id, agent_name, action_type, action_args (y compris les informations de contexte)
        - new_last_rowid : Nouvelle valeur maximale de rowid
    """
    actions = []
    new_last_rowid = last_rowid
    
    if not os.path.exists(db_path):
        return actions, new_last_rowid
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Utiliser rowid pour suivre les enregistrements traités (rowid est le champ auto-incrémenté intégré de SQLite)
        # Ceci évite les différences de format created_at (Twitter utilise des entiers, Reddit utilise des chaînes datetime)
        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))
        
        for rowid, user_id, action, info_json in cursor.fetchall():
            # Mettre à jour le rowid maximal
            new_last_rowid = rowid
            
            # Filtrer les actions non essentielles
            if action in FILTERED_ACTIONS:
                continue
            
            # Analyser les arguments de l'action
            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}
            
            # Simplifier action_args, ne garder que les champs clés (conserver le contenu complet, pas de troncature)
            simplified_args = {}
            if 'content' in action_args:
                simplified_args['content'] = action_args['content']
            if 'post_id' in action_args:
                simplified_args['post_id'] = action_args['post_id']
            if 'comment_id' in action_args:
                simplified_args['comment_id'] = action_args['comment_id']
            if 'quoted_id' in action_args:
                simplified_args['quoted_id'] = action_args['quoted_id']
            if 'new_post_id' in action_args:
                simplified_args['new_post_id'] = action_args['new_post_id']
            if 'follow_id' in action_args:
                simplified_args['follow_id'] = action_args['follow_id']
            if 'query' in action_args:
                simplified_args['query'] = action_args['query']
            if 'like_id' in action_args:
                simplified_args['like_id'] = action_args['like_id']
            if 'dislike_id' in action_args:
                simplified_args['dislike_id'] = action_args['dislike_id']
            
            # Convertir les noms de types d'actions
            action_type = ACTION_TYPE_MAP.get(action, action.upper())
            
            # Compléter les informations de contexte (contenu des publications, noms d'utilisateurs, etc.)
            _enrich_action_context(cursor, action_type, simplified_args, agent_names)
            
            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified_args,
            })
        
        conn.close()
    except Exception as e:
        print(f"Échec de la lecture des actions de la base de données : {e}")
    
    return actions, new_last_rowid


def _enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str]
) -> None:
    """
    Compléter les informations de contexte pour les actions (contenu des publications, noms d'utilisateurs, etc.)
    
    Args :
        cursor : Curseur de base de données
        action_type : Type d'action
        action_args : Arguments de l'action (sera modifié)
        agent_names : Correspondance agent_id -> agent_name
    """
    try:
        # Like/dislike publication : compléter le contenu de la publication et l'auteur
        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
        
        # Repost : compléter le contenu de la publication originale et l'auteur
        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:
                # Le original_post_id du repost pointe vers la publication originale
                cursor.execute("""
                    SELECT original_post_id FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(cursor, original_post_id, agent_names)
                    if original_info:
                        action_args['original_content'] = original_info.get('content', '')
                        action_args['original_author_name'] = original_info.get('author_name', '')
        
        # Citer une publication : compléter le contenu de la publication originale, l'auteur et le commentaire de citation
        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')
            
            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args['original_content'] = original_info.get('content', '')
                    action_args['original_author_name'] = original_info.get('author_name', '')
            
            # Obtenir le contenu du commentaire de la publication citée (quote_content)
            if new_post_id:
                cursor.execute("""
                    SELECT quote_content FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]
        
        # Suivre un utilisateur : compléter le nom de l'utilisateur suivi
        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:
                # Obtenir followee_id depuis la table follow
                cursor.execute("""
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """, (follow_id,))
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args['target_user_name'] = target_name
        
        # Mettre en sourdine un utilisateur : compléter le nom de l'utilisateur mis en sourdine
        elif action_type == 'MUTE':
            # Obtenir user_id ou target_id depuis action_args
            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args['target_user_name'] = target_name
        
        # Like/dislike commentaire : compléter le contenu du commentaire et l'auteur
        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args['comment_content'] = comment_info.get('content', '')
                    action_args['comment_author_name'] = comment_info.get('author_name', '')
        
        # Publier un commentaire : compléter les informations de la publication commentée
        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')
    
    except Exception as e:
        # L'échec de la complémentation du contexte n'affecte pas le processus principal
        print(f"Échec de la complémentation du contexte de l'action : {e}")


def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Obtenir les informations d'une publication
    
    Args :
        cursor : Curseur de base de données
        post_id : ID de la publication
        agent_names : Correspondance agent_id -> agent_name
        
    Retourne :
        Dictionnaire contenant content et author_name, ou None
    """
    try:
        cursor.execute("""
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """, (post_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            
            # Utiliser de préférence le nom de agent_names
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Obtenir le nom depuis la table user
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str]
) -> Optional[str]:
    """
    Obtenir le nom d'un utilisateur
    
    Args :
        cursor : Curseur de base de données
        user_id : ID de l'utilisateur
        agent_names : Correspondance agent_id -> agent_name
        
    Retourne :
        Nom de l'utilisateur, ou None
    """
    try:
        cursor.execute("""
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]
            
            # Utiliser de préférence le nom de agent_names
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except Exception:
        pass
    return None


def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Obtenir les informations d'un commentaire
    
    Args :
        cursor : Curseur de base de données
        comment_id : ID du commentaire
        agent_names : Correspondance agent_id -> agent_name
        
    Retourne :
        Dictionnaire contenant content et author_name, ou None
    """
    try:
        cursor.execute("""
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """, (comment_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]
            
            # Utiliser de préférence le nom de agent_names
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Obtenir le nom depuis la table user
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''
            
            return {'content': content, 'author_name': author_name}
    except Exception:
        pass
    return None


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    Créer le modèle LLM
    
    Support de la configuration double LLM pour l'accélération pendant la simulation parallèle :
    - Configuration commune : LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    - Configuration d'accélération (optionnelle) : LLM_BOOST_API_KEY, LLM_BOOST_BASE_URL, LLM_BOOST_MODEL_NAME
    
    Si le LLM d'accélération est configuré, différentes plateformes peuvent utiliser différents fournisseurs d'API pendant la simulation parallèle pour améliorer la concurrence.
    
    Args :
        config : Dictionnaire de configuration de simulation
        use_boost : S'il faut utiliser la configuration LLM d'accélération (si disponible)
    """
    # Vérifier si la configuration d'accélération existe
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)
    
    # Choisir quel LLM utiliser en fonction des paramètres et de la configuration
    if use_boost and has_boost_config:
        # Utiliser la configuration d'accélération
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM d'accélération]"
    else:
        # Utiliser la configuration commune
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM commun]"
    
    # Si le nom du modèle n'est pas dans .env, utiliser la configuration comme solution de secours
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    
    # Définir les variables d'environnement requises par camel-ai
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Configuration de clé API manquante, veuillez définir LLM_API_KEY dans le fichier .env à la racine du projet")
    
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    print(f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'default'}...")
    
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """Décider quels Agents activer ce tour en fonction de l'heure et de la configuration"""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])
    
    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    
    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0
    
    target_count = int(random.uniform(base_min, base_max) * multiplier)
    
    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        
        if current_hour not in active_hours:
            continue
        
        if random.random() < activity_level:
            candidates.append(agent_id)
    
    selected_ids = random.sample(
        candidates, 
        min(target_count, len(candidates))
    ) if candidates else []
    
    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    
    return active_agents


class PlatformSimulation:
    """Conteneur de résultat de simulation par plateforme"""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Exécuter la simulation Twitter
    
    Args :
        config : Configuration de simulation
        simulation_dir : Répertoire de simulation
        action_logger : Journaliseur d'actions
        main_logger : Gestionnaire de journal principal
        max_rounds : Nombre maximum de tours de simulation (optionnel, utilisé pour tronquer les longues simulations)
        
    Retourne :
        PlatformSimulation : Objet résultat contenant env et agent_graph
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")
    
    log_info("Initialisation...")
    
    # Twitter utilise la configuration LLM commune
    model = create_model(config, use_boost=False)
    
    # OASIS Twitter utilise le format CSV
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"Erreur : Le fichier de profil n'existe pas : {profile_path}")
        return result
    
    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    
    # Obtenir la correspondance des vrais noms d'Agents depuis la config (utiliser entity_name au lieu du Agent_X par défaut)
    agent_names = get_agent_names_from_config(config)
    # Si un agent n'est pas dans la config, utiliser le nom par défaut OASIS
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=30,  # Limiter le nombre maximal de requêtes LLM simultanées pour éviter la surcharge de l'API
    )
    
    await result.env.reset()
    log_info("Environnement démarré")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # Suivre la dernière ligne traitée dans la base de données (utiliser rowid pour éviter les différences de format created_at)
    
    # Exécuter les événements initiaux
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # Enregistrer le début du tour 0 (phase d'événements initiaux)
    if action_logger:
        action_logger.log_round_start(0, 0)  # tour 0, heure_simulée 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"{len(initial_actions)} publications initiales publiées")
    
    # Enregistrer la fin du tour 0
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # Boucle de simulation principale
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # Si le nombre maximum de tours est spécifié, tronquer
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Tours tronqués : {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        # Vérifier si un signal de sortie a été reçu
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Signal de sortie reçu, arrêt de la simulation au tour {round_num + 1}")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        # Enregistrer le début du tour indépendamment des agents actifs
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            # Enregistrer la fin du tour même sans agents actifs (actions_count=0)
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)
        
        # Obtenir les actions réellement exécutées depuis la base de données et les enregistrer
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    # Note : Ne pas fermer l'environnement, le conserver pour l'utilisation Interview
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Boucle de simulation terminée ! Temps écoulé : {elapsed:.1f} secondes, Actions totales : {total_actions}")
    
    return result


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """Exécuter la simulation Reddit
    
    Args :
        config : Configuration de simulation
        simulation_dir : Répertoire de simulation
        action_logger : Journaliseur d'actions
        main_logger : Gestionnaire de journal principal
        max_rounds : Nombre maximum de tours de simulation (optionnel, utilisé pour tronquer les longues simulations)
        
    Retourne :
        PlatformSimulation : Objet résultat contenant env et agent_graph
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")
    
    log_info("Initialisation...")
    
    # Reddit utilise la configuration LLM d'accélération (si disponible, sinon utilise la configuration commune)
    model = create_model(config, use_boost=True)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Erreur : Le fichier de profil n'existe pas : {profile_path}")
        return result
    
    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    
    # Obtenir la correspondance des vrais noms d'Agents depuis la config (utiliser entity_name au lieu du Agent_X par défaut)
    agent_names = get_agent_names_from_config(config)
    # Si un agent n'est pas dans la config, utiliser le nom par défaut OASIS
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, 'name', f'Agent_{agent_id}')
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=30,  # Limiter le nombre maximal de requêtes LLM simultanées pour éviter la surcharge de l'API
    )
    
    await result.env.reset()
    log_info("Environnement démarré")
    
    if action_logger:
        action_logger.log_simulation_start(config)
    
    total_actions = 0
    last_rowid = 0  # Suivre la dernière ligne traitée dans la base de données (utiliser rowid pour éviter les différences de format created_at)
    
    # Exécuter les événements initiaux
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    # Enregistrer le début du tour 0 (phase d'événements initiaux)
    if action_logger:
        action_logger.log_round_start(0, 0)  # tour 0, heure_simulée 0
    
    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                if agent in initial_actions:
                    if not isinstance(initial_actions[agent], list):
                        initial_actions[agent] = [initial_actions[agent]]
                    initial_actions[agent].append(ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    ))
                else:
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content}
                    )
                
                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content}
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"{len(initial_actions)} publications initiales publiées")
    
    # Enregistrer la fin du tour 0
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)
    
    # Boucle de simulation principale
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # Si le nombre maximum de tours est spécifié, tronquer
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"Tours tronqués : {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()
    
    for round_num in range(total_rounds):
        # Vérifier si un signal de sortie a été reçu
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"Signal de sortie reçu, arrêt de la simulation au tour {round_num + 1}")
            break
        
        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        # Enregistrer le début du tour indépendamment des agents actifs
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)
        
        if not active_agents:
            # Enregistrer la fin du tour même sans agents actifs (actions_count=0)
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue
        
        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)
        
        # Obtenir les actions réellement exécutées depuis la base de données et les enregistrer
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )
        
        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data['agent_id'],
                    agent_name=action_data['agent_name'],
                    action_type=action_data['action_type'],
                    action_args=action_data['action_args']
                )
                total_actions += 1
                round_action_count += 1
        
        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)
        
        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)")
    
    # Note : Ne pas fermer l'environnement, le conserver pour l'utilisation Interview
    
    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)
    
    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(f"Boucle de simulation terminée ! Temps écoulé : {elapsed:.1f} secondes, Actions totales : {total_actions}")
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='Simulation parallèle double plateforme OASIS')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Chemin du fichier de configuration (simulation_config.json)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='Exécuter uniquement la simulation Twitter'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='Exécuter uniquement la simulation Reddit'
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
        help="Fermer l'environnement immédiatement après la fin de la simulation, ne pas entrer en mode d'attente"
    )
    
    args = parser.parse_args()
    
    # Créer l'événement d'arrêt au début de la fonction main pour assurer que tout le programme puisse répondre au signal de sortie
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"Erreur : Le fichier de configuration n'existe pas : {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    
    # Initialiser la configuration de journalisation (désactiver les journaux OASIS, nettoyer les anciens fichiers)
    init_logging_for_simulation(simulation_dir)
    
    # Créer le gestionnaire de journaux
    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()
    
    log_manager.info("=" * 60)
    log_manager.info("Simulation parallèle double plateforme OASIS")
    log_manager.info(f"Fichier de configuration : {args.config}")
    log_manager.info(f"ID de simulation : {config.get('simulation_id', 'inconnu')}")
    log_manager.info(f"Mode d'attente : {'Activé' if wait_for_commands else 'Désactivé'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info(f"Paramètres de simulation :")
    log_manager.info(f"  - Durée totale de simulation : {total_hours} heures")
    log_manager.info(f"  - Temps par tour : {minutes_per_round} minutes")
    log_manager.info(f"  - Tours totaux configurés : {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - Limite maximale de tours : {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - Tours réellement exécutés : {args.max_rounds} (Tronqué)")
    log_manager.info(f"  - Nombre d'Agents : {len(config.get('agent_configs', []))}")
    
    log_manager.info("Structure des journaux :")
    log_manager.info(f"  - Journal principal : simulation.log")
    log_manager.info(f"  - Actions Twitter : twitter/actions.jsonl")
    log_manager.info(f"  - Actions Reddit : reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    
    # Stocker les résultats de simulation des deux plateformes
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    if args.twitter_only:
        twitter_result = await run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds)
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds)
    else:
        # Exécuter en parallèle (chaque plateforme utilise un journaliseur indépendant)
        results = await asyncio.gather(
            run_twitter_simulation(config, simulation_dir, twitter_logger, log_manager, args.max_rounds),
            run_reddit_simulation(config, simulation_dir, reddit_logger, log_manager, args.max_rounds),
        )
        twitter_result, reddit_result = results
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"Boucle de simulation terminée ! Temps total : {total_elapsed:.1f} secondes")
    
    # S'il faut entrer en mode d'attente
    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("Entrée en mode d'attente - l'environnement reste en cours d'exécution")
        log_manager.info("Commandes supportées : interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        
        # Créer le gestionnaire IPC
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None
        )
        ipc_handler.update_status("alive")
        
        # Boucle d'attente de commandes (utilisant le _shutdown_event global)
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                # Utiliser wait_for au lieu de sleep pour répondre au shutdown_event
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # Signal de sortie reçu
                except asyncio.TimeoutError:
                    pass  # Délai dépassé, continuer la boucle
        except KeyboardInterrupt:
            print("\nSignal d'interruption reçu")
        except asyncio.CancelledError:
            print("\nTâche annulée")
        except Exception as e:
            print(f"\nErreur lors du traitement de la commande : {e}")
        
        log_manager.info("\nFermeture de l'environnement...")
        ipc_handler.update_status("stopped")
    
    # Fermer l'environnement
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] Environnement fermé")
    
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] Environnement fermé")
    
    log_manager.info("=" * 60)
    log_manager.info(f"Tout est terminé !")
    log_manager.info(f"Fichiers journaux :")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)


def setup_signal_handlers(loop=None):
    """
    Configurer les gestionnaires de signaux pour assurer une sortie propre lors de la réception de SIGTERM/SIGINT
    
    Scénario de simulation persistante : La simulation terminée ne quitte pas, attend les commandes d'interview
    Lors de la réception d'un signal de terminaison, il faut :
    1. Notifier la boucle asyncio de quitter l'attente
    2. Donner au programme une chance de nettoyer correctement les ressources (fermer la base de données, l'environnement, etc.)
    3. Puis quitter
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nSignal {sig_name} reçu, fermeture en cours...")
        
        if not _cleanup_done:
            _cleanup_done = True
            # Définir l'événement pour notifier la boucle asyncio de quitter (donner à la boucle une chance de nettoyer)
            if _shutdown_event:
                _shutdown_event.set()
        
        # Ne pas appeler sys.exit() directement, laisser la boucle asyncio quitter normalement et nettoyer
        # Forcer la sortie uniquement si le signal est reçu de manière répétée
        else:
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
        # Nettoyer le traqueur de ressources multiprocessing (prévient les avertissements à la sortie)
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Processus de simulation terminé")
