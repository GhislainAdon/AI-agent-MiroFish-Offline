"""
Point d'entrée du backend MiroFish
"""

import os
import sys

# Résoudre le problème d'encodage des caractères chinois dans la console Windows : définir l'encodage UTF-8 avant toutes les importations
if sys.platform == 'win32':
    # Définir la variable d'environnement pour s'assurer que Python utilise UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigurer le flux de sortie standard en UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ajouter le répertoire racine du projet au chemin
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Fonction principale"""
    # Valider la configuration
    errors = Config.validate()
    if errors:
        print("Erreurs de configuration :")
        for err in errors:
            print(f"  - {err}")
        print("\nVeuillez vérifier la configuration dans le fichier .env")
        sys.exit(1)

    # Créer l'application
    app = create_app()

    # Obtenir la configuration d'exécution
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG

    # Démarrer le service
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
