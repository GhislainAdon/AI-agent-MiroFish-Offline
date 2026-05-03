"""
Gestion de la configuration
Charge la configuration depuis le fichier .env à la racine du projet
"""

import os
from dotenv import load_dotenv

# Charger le fichier .env depuis la racine du projet
# Chemin : MiroFish/.env (relatif à backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # Si pas de .env à la racine, essayer de charger les variables d'environnement (pour la production)
    load_dotenv(override=True)


class Config:
    """Classe de configuration Flask"""

    # Configuration Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # Configuration JSON - désactiver l'échappement ASCII pour afficher le chinois directement (pas comme \uXXXX)
    JSON_AS_ASCII = False

    # Configuration LLM (format OpenAI unifié)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'qwen2.5:32b')

    # Configuration Neo4j
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'mirofish')

    # Configuration d'embedding
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:11434')

    # Configuration de téléchargement de fichiers
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50Mo
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Configuration du traitement de texte
    DEFAULT_CHUNK_SIZE = 500  # Taille de morceau par défaut
    DEFAULT_CHUNK_OVERLAP = 50  # Taille de chevauchement par défaut

    # Configuration de simulation OASIS
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # Configuration des actions disponibles sur les plateformes OASIS
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Configuration de l'agent de rapport
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        """Valider la configuration requise"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY non configuré (définir à toute valeur non vide, par ex. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI non configuré")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD non configuré")
        return errors
