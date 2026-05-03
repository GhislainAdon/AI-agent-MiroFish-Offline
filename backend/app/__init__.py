"""
MiroFish Backend - Fabrique d'application Flask
"""

import os
import warnings

# Supprimer les avertissements multiprocessing resource_tracker (des bibliothèques tierces comme transformers)
# Doit être défini avant toutes les autres importations
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Fonction fabrique de l'application Flask"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configurer l'encodage JSON : s'assurer que le chinois s'affiche directement (pas comme \uXXXX)
    # Flask >= 2.3 utilise app.json.ensure_ascii, les versions plus anciennes utilisent la config JSON_AS_ASCII
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False

    # Configuration de la journalisation
    logger = setup_logger('mirofish')

    # N'imprimer les informations de démarrage que dans le sous-processus du reloader (éviter d'imprimer deux fois en mode debug)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("Démarrage du backend MiroFish-Offline...")
        logger.info("=" * 50)

    # Activer CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # --- Initialiser le singleton Neo4jStorage (injection via app.extensions) ---
    from .storage import Neo4jStorage
    try:
        neo4j_storage = Neo4jStorage()
        app.extensions['neo4j_storage'] = neo4j_storage
        if should_log_startup:
            logger.info("Neo4jStorage initialisé (connecté à %s)", Config.NEO4J_URI)
    except Exception as e:
        logger.error("Échec de l'initialisation de Neo4jStorage : %s", e)
        # Stocker None pour que les endpoints puissent retourner 503 proprement
        app.extensions['neo4j_storage'] = None

    # Enregistrer la fonction de nettoyage des processus de simulation (s'assurer que tous les processus de simulation se terminent à l'arrêt du serveur)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Fonction de nettoyage des processus de simulation enregistrée")

    # Middleware de journalisation des requêtes
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Requête : {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Corps de la requête : {request.get_json(silent=True)}")

    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Réponse : {response.status_code}")
        return response

    # Enregistrer les blueprints
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')

    # Vérification de l'état de santé
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish-Offline Backend'}

    if should_log_startup:
        logger.info("Démarrage du backend MiroFish-Offline terminé")

    return app
