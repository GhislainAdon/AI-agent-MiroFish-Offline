"""
Module de configuration du journaliseur
Fournit une gestion unifiée de la journalisation avec sortie vers la console et un fichier
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """
    S'assurer que stdout/stderr utilisent l'encodage UTF-8
    Résout le problème d'encodage des caractères chinois dans la console Windows
    """
    if sys.platform == 'win32':
        # Reconfigurer la sortie standard en UTF-8 sous Windows
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Répertoire des journaux
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


def setup_logger(name: str = 'mirofish', level: int = logging.DEBUG) -> logging.Logger:
    """
    Configurer le journaliseur

    Args:
        name: Nom du journaliseur
        level: Niveau de journalisation

    Returns:
        Journaliseur configuré
    """
    # S'assurer que le répertoire des journaux existe
    os.makedirs(LOG_DIR, exist_ok=True)

    # Créer le journaliseur
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Empêcher la propagation des journaux vers le journaliseur racine pour éviter les sorties en double
    logger.propagate = False

    # Si des gestionnaires existent déjà, ne pas ajouter de doublons
    if logger.handlers:
        return logger

    # Formats de journalisation
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # 1. Gestionnaire de fichier - journaux détaillés (nommés par date, avec rotation)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10 Mo
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # 2. Gestionnaire de console - journaux concis (INFO et au-dessus)
    # S'assurer de l'encodage UTF-8 sous Windows pour éviter les problèmes de caractères chinois
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Ajouter les gestionnaires
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = 'mirofish') -> logging.Logger:
    """
    Obtenir le journaliseur (le créer s'il n'existe pas)

    Args:
        name: Nom du journaliseur

    Returns:
        Instance du journaliseur
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Créer le journaliseur par défaut
logger = setup_logger()


# Fonctions de commodité
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
