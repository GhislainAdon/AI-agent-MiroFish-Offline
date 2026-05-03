"""
Routes API liées à la simulation.
Étape 2 : lecture et filtrage des entités, préparation OASIS et exécution automatisée.
"""

import os
import traceback
from flask import request, jsonify, send_file, current_app

from . import simulation_bp
from ..config import Config
from ..services.entity_reader import EntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..utils.logger import get_logger
from ..models.project import ProjectManager

logger = get_logger('mirofish.api.simulation')


# Préfixe d'optimisation des prompts d'entretien.
# Il force les agents à répondre directement en texte au lieu d'appeler des outils.
INTERVIEW_PROMPT_PREFIX = "Based on your persona, all your past memories and actions, reply directly to me with text without calling any tools:"


def optimize_interview_prompt(prompt: str) -> str:
    """
    Optimiser les questions d'entretien en ajoutant un préfixe qui évite les appels d'outils.
    
    Args:
        prompt: Question originale
        
    Returns:
        Question optimisée
    """
    if not prompt:
        return prompt
    # Éviter d'ajouter le préfixe plusieurs fois
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Interface de lecture des entités ==============

@simulation_bp.route('/entities/<graph_id>', methods=['GET'])
def get_graph_entities(graph_id: str):
    """
    Obtenir toutes les entités du graphe de connaissances (filtrées)
    
    Retourne uniquement les nœuds correspondant aux types d'entités prédéfinis (nœuds dont les Labels ne sont pas seulement Entity)
    
    Paramètres de requête :
        entity_types : liste de types d'entités séparés par des virgules (facultatif, pour un filtrage supplémentaire)
        enrich : whether pour obtenir les informations d'arêtes associées (défaut true)
    """
    try:
        entity_types_str = request.args.get('entity_types', '')
        entity_types = [t.strip() for t in entity_types_str.split(',') if t.strip()] if entity_types_str else None
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        logger.info(f"Obtention des entités du graphe de connaissances : graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}")
        
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé")
        reader = EntityReader(storage)
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des entités du graphe de connaissances : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/<entity_uuid>', methods=['GET'])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Obtenir les informations détaillées d'une seule entité"""
    try:
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé")
        reader = EntityReader(storage)
        entity = reader.get_entity_with_context(graph_id, entity_uuid)
        
        if not entity:
            return jsonify({
                "success": False,
                "error": f"L'entité n'existe pas : {entity_uuid}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": entity.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des détails de l'entité : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/entities/<graph_id>/by-type/<entity_type>', methods=['GET'])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Obtenir toutes les entités du type spécifié"""
    try:
        enrich = request.args.get('enrich', 'true').lower() == 'true'
        
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé")
        reader = EntityReader(storage)
        entities = reader.get_entities_by_type(
            graph_id=graph_id,
            entity_type=entity_type,
            enrich_with_edges=enrich
        )
        
        return jsonify({
            "success": True,
            "data": {
                "entity_type": entity_type,
                "count": len(entities),
                "entities": [e.to_dict() for e in entities]
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des entités : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de gestion de simulation ==============

@simulation_bp.route('/create', methods=['POST'])
def create_simulation():
    """
    Créer une nouvelle simulation.
    
    Note : les paramètres comme max_rounds sont intelligemment générés par le LLM, aucun réglage manuel nécessaire
    
    Requête (JSON) :
        {
            "project_id": "proj_xxxx",      // Requis
            "graph_id": "mirofish_xxxx",    // Facultatif, si non fourni, obtenu depuis le projet
            "enable_twitter": true,          // Facultatif, défaut true
            "enable_reddit": true            // Facultatif, défaut true
        }
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir project_id"
            }), 400
        
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Le projet n'existe pas : {project_id}"
            }), 404
        
        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Le projet n'a pas encore construit de graphe de connaissances, veuillez d'abord appeler /api/graph/build"
            }), 400
        
        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )
        
        return jsonify({
            "success": True,
            "data": state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Échec de la création de la simulation : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Vérifier si la simulation est prête.
    
    Conditions vérifiées :
    1. state.json existe et le statut est "ready".
    2. Les fichiers requis existent : reddit_profiles.json, twitter_profiles.csv, simulation_config.json.
    
    Note : les scripts d'exécution (run_*.py) restent dans le répertoire backend/scripts/, ils ne sont plus copiés dans le répertoire de simulation
    
    Args:
        simulation_id: ID de la simulation
        
    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config
    
    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
    
    # Vérifier si le répertoire existe
    if not os.path.exists(simulation_dir):
        return False, {"reason": "Le répertoire de simulation n'existe pas"}
    
    # Liste des fichiers requis (scripts non inclus, les scripts sont dans backend/scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv"
    ]
    
    # Vérifier si les fichiers existent
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)
    
    if missing_files:
        return False, {
            "reason": "Fichiers requis manquants",
            "missing_files": missing_files,
            "existing_files": existing_files
        }
    
    # Vérifier le statut dans state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)
        
        # Journaux détaillés
        logger.debug(f"Détection du statut de préparation de la simulation : {simulation_id}, status={status}, config_generated={config_generated}")
        
        # Si config_generated=True et les fichiers existent, considérer la préparation comme terminée
        # Les statuts suivants indiquent que la préparation est terminée :
        # - ready : Préparation terminée, peut être exécutée
        # - preparing : Si config_generated=True, la description indique que c'est terminé
        # - running : En cours d'exécution, préparation déjà terminée
        # - completed : Exécution terminée, préparation déjà terminée
        # - stopped : Arrêté, préparation déjà terminée
        # - failed : Échec de l'exécution (mais la préparation est terminée)
        prepared_statuses = ["ready", "preparing", "running", "completed", "stopped", "failed"]
        if status in prepared_statuses and config_generated:
            # Obtenir les statistiques des fichiers
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")
            
            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                    profiles_count = len(profiles_data) if isinstance(profiles_data, list) else 0
            
            # Si le statut est "preparing" mais les fichiers sont complets, mettre à jour le statut à "ready"
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime
                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Mise à jour automatique du statut de simulation : {simulation_id} preparing -> ready")
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Échec de la mise à jour automatique du statut : {e}")
            
            logger.info(f"Simulation {simulation_id} Résultat de détection : Préparation terminée (status={status}, config_generated={config_generated})")
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files
            }
        else:
            logger.warning(f"Simulation {simulation_id} Résultat de détection : Préparation non terminée (status={status}, config_generated={config_generated})")
            return False, {
                "reason": f"Le statut n'est pas dans la liste des préparés ou config_generated est faux : status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated
            }
            
    except Exception as e:
        return False, {"reason": f"Échec de la lecture du fichier d'état : {str(e)}"}


@simulation_bp.route('/prepare', methods=['POST'])
def prepare_simulation():
    """
    Préparer l'environnement de simulation avec une tâche asynchrone et une configuration générée par LLM.

    Opération longue : l'interface retourne immédiatement un task_id.
    Utiliser /api/simulation/prepare/status pour suivre la progression.

    Fonctionnalités :
    - Détecte les préparations déjà terminées pour éviter les doublons.
    - Retourne directement les résultats existants si la préparation est déjà faite.
    - Supporte la régénération forcée avec force_regenerate=true.

    Étapes :
    1. Vérifier si la préparation est déjà terminée.
    2. Lire et filtrer les entités du graphe de connaissances.
    3. Générer un profil agent OASIS pour chaque entité.
    4. Générer la configuration OASIS via LLM.
    5. Enregistrer les fichiers de configuration et les profils.
    
    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",                   // Requis, ID de simulation
            "entity_types": ["Student", "PublicFigure"],  // Facultatif, type d'entité spécifié
            "use_llm_for_profiles": true,                 // Facultatif, utiliser ou non le LLM pour générer les personas
            "parallel_profile_count": 5,                  // Facultatif, nombre de personas à générer en parallèle, défaut 5
            "force_regenerate": false                     // Facultatif, forcer la génération, défaut false
        }
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // Retourné pour les nouvelles tâches
                "status": "preparing|ready",
                "message": "Tâche de préparation démarrée|Préparation déjà terminée",
                "already_prepared": true|false    // La préparation est-elle terminée
            }
        }
    """
    import threading
    import os
    from ..models.task import TaskManager, TaskStatus
    from ..config import Config
    
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400
        
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"La simulation n'existe pas : {simulation_id}"
            }), 404
        
        # Vérifier si la régénération est forcée
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"Début du traitement de la requête /prepare : simulation_id={simulation_id}, force_regenerate={force_regenerate}")
        
        # Vérifier si déjà préparé (éviter les générations en double)
        if not force_regenerate:
            logger.debug(f"Vérification si la simulation {simulation_id} est préparée...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(f"Résultat de la vérification : is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"La simulation {simulation_id} a déjà été préparée, pas besoin de régénérer")
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": "Préparation déjà terminée, pas besoin de régénérer",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
            else:
                logger.info(f"La simulation {simulation_id} n'est pas encore préparée, préparation en cours")
        
        # Obtenir les informations nécessaires depuis le projet
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Le projet n'existe pas : {state.project_id}"
            }), 404
        
        # Obtenir les exigences de simulation
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Description de l'exigence de simulation manquante dans le projet (simulation_requirement)"
            }), 400
        
        # Obtenir le texte du document
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""
        
        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = data.get('parallel_profile_count', 5)
        
        # ========== Obtenir GraphStorage (capturer la référence avant le démarrage de la tâche en arrière-plan) ==========
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé — vérifiez la connexion Neo4j")

        # ========== Obtenir synchroniquement le nombre d'entités (avant le démarrage de la tâche en arrière-plan) ==========
        # Ainsi le frontend peut obtenir immédiatement le nombre total d'agents attendu lors de l'appel à /prepare
        try:
            logger.info(f"Obtention synchronique du nombre d'entités : graph_id={state.graph_id}")
            reader = EntityReader(storage)
            # Lire rapidement les entités (sans informations d'arêtes, seules les statistiques sont nécessaires)
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False  # Pas d'informations d'arêtes, accélérer
            )
            # Sauvegarder le nombre d'entités dans le statut (pour que le frontend puisse l'obtenir immédiatement)
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"Nombre d'entités attendu : {filtered_preview.filtered_count}, [type][modèle] : {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"Échec de l'obtention synchronique du nombre d'entités (nouvelle tentative dans la tâche en arrière-plan) : {e}")
            # L'échec n'affecte pas le processus ultérieur, la tâche en arrière-plan réessaiera
        
        # Créer une tâche asynchrone
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )
        
        # Mettre à jour le statut de la simulation (inclure le nombre d'entités pré-récupérées)
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)
        
        # Définir la tâche en arrière-plan
        def run_prepare():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Début de la préparation de l'environnement de simulation..."
                )
                
                # Préparer la simulation (avec rappel de progression)
                # Stocker les détails de progression par étape
                stage_details = {}
                
                def progress_callback(stage, progress, message, **kwargs):
                    # Calculer la progression totale
                    stage_weights = {
                        "reading": (0, 20),           # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),    # 70-90%
                        "copying_scripts": (90, 100)       # 90-100%
                    }
                    
                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)
                    
                    # Construire les informations de progression détaillées
                    stage_names = {
                        "reading": "Lecture des entités du graphe de connaissances",
                        "generating_profiles": "Génération des personas d'agents",
                        "generating_config": "Génération de la configuration de simulation",
                        "copying_scripts": "Préparation des scripts de simulation"
                    }
                    
                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)
                    
                    # Mettre à jour les détails de l'étape
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }
                    
                    # Construire les informations de progression détaillées
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }
                    
                    # Construire un message concis
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)} : "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)} : {message}"
                    
                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )
                
                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count,
                    storage=storage,
                )
                
                # Tâche terminée
                task_manager.complete_task(
                    task_id,
                    result=result_state.to_simple_dict()
                )
                
            except Exception as e:
                logger.error(f"Échec de la préparation de la simulation : {str(e)}")
                task_manager.fail_task(task_id, str(e))
                
                # Mettre à jour le statut de la simulation à échoué
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)
        
        # Démarrer le thread d'arrière-plan
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": "Tâche de préparation démarrée, veuillez suivre la progression via /api/simulation/prepare/status",
                "already_prepared": False,
                "expected_entities_count": state.entities_count,  # Nombre d'entités attendues à traiter
                "entity_types": state.entity_types  # Liste des types d'entités
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Échec du démarrage de la tâche de préparation : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/prepare/status', methods=['POST'])
def get_prepare_status():
    """
    Consulter la progression de la tâche de préparation
    
    Supporte deux méthodes de consultation :
    1. Consulter via task_id pour vérifier la progression de la tâche en cours
    2. Vérifier via simulation_id si la préparation est déjà terminée
    
    Requête (JSON) :
        {
            "task_id": "task_xxxx",          // Facultatif, depuis l'appel /prepare précédent
            "simulation_id": "sim_xxxx"      // Facultatif, ID de simulation (pour vérifier si la préparation est terminée)
        }
    
    Retourne :
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // Y a-t-il une préparation terminée
                "prepare_info": {...}            // Informations détaillées lorsque la préparation est terminée
            }
        }
    """
    from ..models.task import TaskManager
    
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # Si simulation_id est fourni, vérifier si la préparation est terminée
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": "Préparation déjà terminée",
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
        
        # Si pas de task_id, retourner une erreur
        if not task_id:
            if simulation_id:
                # A simulation_id mais préparation non terminée
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": "Préparation pas encore démarrée, veuillez appeler /api/simulation/prepare",
                        "already_prepared": False
                    }
                })
            return jsonify({
                "success": False,
                "error": "Veuillez fournir task_id ou simulation_id"
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            # La tâche n'existe pas, mais si simulation_id est fourni, vérifier si la préparation est terminée
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": "Tâche terminée (préparation déjà existante)",
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    })
            
            return jsonify({
                "success": False,
                "error": f"La tâche n'existe pas : {task_id}"
            }), 404
        
        task_dict = task.to_dict()
        task_dict["already_prepared"] = False
        
        return jsonify({
            "success": True,
            "data": task_dict
        })
        
    except Exception as e:
        logger.error(f"Échec de la consultation du statut de la tâche : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@simulation_bp.route('/<simulation_id>', methods=['GET'])
def get_simulation(simulation_id: str):
    """Obtenir le statut de la simulation"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": f"La simulation n'existe pas : {simulation_id}"
            }), 404
        
        result = state.to_dict()
        
        # Si la simulation est prête, instructions d'exécution supplémentaires
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention du statut de la simulation : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/list', methods=['GET'])
def list_simulations():
    """
    Lister toutes les simulations
    
    Paramètres de requête :
        project_id : Filtrer par ID de projet (facultatif)
    """
    try:
        project_id = request.args.get('project_id')
        
        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)
        
        return jsonify({
            "success": True,
            "data": [s.to_dict() for s in simulations],
            "count": len(simulations)
        })
        
    except Exception as e:
        logger.error(f"Échec du listage des simulations : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    Obtenir le dernier report_id correspondant à la simulation
    
    Parcourir le répertoire des rapports et trouver le rapport correspondant au simulation_id.
    Si plusieurs existent, retourner le plus récent (par horodatage created_at).
    
    Args:
        simulation_id: ID de la simulation
        
    Returns:
        report_id ou None
    """
    import json
    from datetime import datetime
    
    # Chemin du répertoire reports : backend/uploads/reports
    # __file__ est app/api/simulation.py, besoin de remonter deux niveaux vers backend/
    reports_dir = os.path.join(os.path.dirname(__file__), '../../uploads/reports')
    if not os.path.exists(reports_dir):
        return None
    
    matching_reports = []
    
    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue
            
            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue
            
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append({
                        "report_id": meta.get("report_id"),
                        "created_at": meta.get("created_at", ""),
                        "status": meta.get("status", "")
                    })
            except Exception:
                continue
        
        if not matching_reports:
            return None
        
        # Trier par date de création décroissante, retourner le plus récent
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")
        
    except Exception as e:
        logger.warning(f"Échec de la recherche de rapport pour la simulation {simulation_id} : {e}")
        return None


@simulation_bp.route('/history', methods=['GET'])
def get_simulation_history():
    """
    Obtenir la liste des simulations historiques (avec détails du projet)
    
    Pour l'affichage des projets historiques sur la page d'accueil. Retourne le nom du projet et d'autres informations sur la simulation.
    
    Paramètres de requête :
        limit : Limite du nombre de retours (défaut 20)
    
    Retourne :
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "Analyse d'opinion WDU",
                    "simulation_requirement": "Si l'Université de Wuhan publie...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        
        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]
        
        # Enrichir les données de simulation, lecture uniquement depuis les fichiers de simulation
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()
            
            # Obtenir les informations de configuration de la simulation (lire simulation_requirement depuis simulation_config.json)
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get("simulation_requirement", "")
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get("total_simulation_hours", 0)
                # Tours recommandés (valeur de repli)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0) * 60 / 
                    max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0
            
            # Obtenir le statut d'exécution (depuis run_state.json)
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Utiliser les total_rounds définis par l'utilisateur, sinon utiliser les tours recommandés
                sim_dict["total_rounds"] = run_state.total_rounds if run_state.total_rounds > 0 else recommended_rounds
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds
            
            # Obtenir la liste des fichiers du projet associé (au plus 3 éléments)
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, 'files') and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "Fichier inconnu")} 
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []
            
            # Obtenir le report_id associé (trouver le rapport le plus récent de cette simulation)
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)
            
            # Ajouter le numéro de version
            sim_dict["version"] = "v1.0.2"
            
            # Formater la date
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""
            
            enriched_simulations.append(sim_dict)
        
        return jsonify({
            "success": True,
            "data": enriched_simulations,
            "count": len(enriched_simulations)
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des simulations historiques : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles', methods=['GET'])
def get_simulation_profiles(simulation_id: str):
    """
    Obtenir les profils d'agents de la simulation
    
    Paramètres de requête :
        platform : Type de plateforme (reddit/twitter, défaut reddit)
    """
    try:
        platform = request.args.get('platform', 'reddit')
        
        manager = SimulationManager()
        profiles = manager.get_profiles(simulation_id, platform=platform)
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "count": len(profiles),
                "profiles": profiles
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des profils : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/profiles/realtime', methods=['GET'])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Obtenir en temps réel les profils d'agents de la simulation (pour visualisation pendant la génération).

    Différence avec l'endpoint /profiles :
    - Lit le fichier directement, contourne SimulationManager
    - Pour visualisation en temps réel pendant la génération
    - Retourne des métadonnées supplémentaires (comme l'heure de modification du fichier, si la génération est en cours, etc.)
    
    Paramètres de requête :
        platform : Type de plateforme (reddit/twitter, défaut reddit)
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // Total attendu (si disponible)
                "is_generating": true,  // En cours de génération
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime
    
    try:
        platform = request.args.get('platform', 'reddit')
        
        # Obtenir le répertoire de simulation
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"La simulation n'existe pas : {simulation_id}"
            }), 404
        
        # Déterminer le chemin du fichier
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")
        
        # Vérifier si les fichiers existent
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None
        
        if file_exists:
            # Obtenir l'heure de modification du fichier
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                if platform == "reddit":
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Échec de la lecture du fichier de profils : {e}")
                profiles = []
        
        # Vérifier si la génération est en cours (via le champ status de state.json)
        is_generating = False
        total_expected = None
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("entities_count")
            except Exception:
                pass
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "platform": platform,
                "count": len(profiles),
                "total_expected": total_expected,
                "is_generating": is_generating,
                "file_exists": file_exists,
                "file_modified_at": file_modified_at,
                "profiles": profiles
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention en temps réel des profils : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/realtime', methods=['GET'])
def get_simulation_config_realtime(simulation_id: str):
    """
    Obtenir en temps réel la configuration de la simulation (pour visualisation pendant la génération).

    Différence avec l'endpoint /config :
    - Lit le fichier directement, contourne SimulationManager
    - Pour visualisation en temps réel pendant la génération
    - Retourne des métadonnées supplémentaires (comme l'heure de modification du fichier, si la génération est en cours, etc.)
    - Retourne des informations partielles même si la configuration n'est pas entièrement générée
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // En cours de génération
                "generation_stage": "generating_config",  // Étape de génération actuelle
                "config": {...}  // Contenu de la configuration (si existant)
            }
        }
    """
    import json
    from datetime import datetime
    
    try:
        # Obtenir le répertoire de simulation
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)
        
        if not os.path.exists(sim_dir):
            return jsonify({
                "success": False,
                "error": f"La simulation n'existe pas : {simulation_id}"
            }), 404
        
        # Chemin du fichier de configuration
        config_file = os.path.join(sim_dir, "simulation_config.json")
        
        # Vérifier si les fichiers existent
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None
        
        if file_exists:
            # Obtenir l'heure de modification du fichier
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Échec de la lecture du fichier de configuration : {e}")
                config = None
        
        # Vérifier si la génération est en cours (via le champ status de state.json)
        is_generating = False
        generation_stage = None
        config_generated = False
        
        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    config_generated = state_data.get("config_generated", False)
                    
                    # Déterminer l'étape actuelle
                    if is_generating:
                        if state_data.get("profiles_generated", False):
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
            except Exception:
                pass
        
        # Construire les données de retour
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config
        }
        
        # Si la configuration existe, extraire les statistiques clés
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get("total_simulation_hours"),
                "initial_posts_count": len(config.get("event_config", {}).get("initial_posts", [])),
                "hot_topics_count": len(config.get("event_config", {}).get("hot_topics", [])),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model")
            }
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention en temps réel de la configuration : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config', methods=['GET'])
def get_simulation_config(simulation_id: str):
    """
    Obtenir la configuration de la simulation (générée avec l'intelligence LLM).

    Retourne :
        - time_config : Configuration temporelle (durée de simulation, heure de début, heure de fin, etc.)
        - agent_configs : Configuration d'activité pour chaque agent (comportements, styles d'interaction, etc.)
        - event_config : Configuration des événements (posts initiaux, séquences d'événements, etc.)
        - platform_configs : Configuration de la plateforme
        - generation_reasoning : Explication du raisonnement de configuration LLM
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)
        
        if not config:
            return jsonify({
                "success": False,
                "error": f"La configuration de simulation n'existe pas. Veuillez d'abord appeler /prepare"
            }), 404
        
        return jsonify({
            "success": True,
            "data": config
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention de la configuration : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/config/download', methods=['GET'])
def download_simulation_config(simulation_id: str):
    """Télécharger le fichier de configuration de simulation"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return jsonify({
                "success": False,
                "error": "Le fichier de configuration n'existe pas. Veuillez d'abord appeler /prepare"
            }), 404
        
        return send_file(
            config_path,
            as_attachment=True,
            download_name="simulation_config.json"
        )
        
    except Exception as e:
        logger.error(f"Échec du téléchargement de la configuration : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/script/<script_name>/download', methods=['GET'])
def download_simulation_script(script_name: str):
    """
    Télécharger le fichier de script d'exécution de simulation (scripts généraux depuis backend/scripts/)
    
    Valeurs possibles pour script_name :
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Les scripts se trouvent dans le répertoire backend/scripts/
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        # Vérifier le nom du script
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py", 
            "run_parallel_simulation.py",
            "action_logger.py"
        ]
        
        if script_name not in allowed_scripts:
            return jsonify({
                "success": False,
                "error": f"Script inconnu : {script_name}, options : {allowed_scripts}"
            }), 400
        
        script_path = os.path.join(scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            return jsonify({
                "success": False,
                "error": f"Le fichier de script n'existe pas : {script_name}"
            }), 404
        
        return send_file(
            script_path,
            as_attachment=True,
            download_name=script_name
        )
        
    except Exception as e:
        logger.error(f"Échec du téléchargement du script : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de génération de profils (utilisation autonome) ==============

@simulation_bp.route('/generate-profiles', methods=['POST'])
def generate_profiles():
    """
    Générer directement depuis le graphe de connaissances les profils d'agents OASIS (sans créer de simulation)
    
    Requête (JSON) :
        {
            "graph_id": "mirofish_xxxx",     // Requis
            "entity_types": ["Student"],      // Facultatif
            "use_llm": true,                  // Facultatif
            "platform": "reddit"              // Facultatif
        }
    """
    try:
        data = request.get_json() or {}
        
        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir graph_id"
            }), 400
        
        entity_types = data.get('entity_types')
        use_llm = data.get('use_llm', True)
        platform = data.get('platform', 'reddit')
        
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé")
        reader = EntityReader(storage)
        filtered = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=True
        )
        
        if filtered.filtered_count == 0:
            return jsonify({
                "success": False,
                "error": "Aucune entité correspondante trouvée"
            }), 400
        
        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities,
            use_llm=use_llm
        )
        
        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "entity_types": list(filtered.entity_types),
                "count": len(profiles_data),
                "profiles": profiles_data
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de la génération des profils : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de contrôle de l'exécution de la simulation ==============

@simulation_bp.route('/start', methods=['POST'])
def start_simulation():
    """
    Démarrer l'exécution de la simulation

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",          // Requis, ID de simulation
            "platform": "parallel",                // Facultatif : twitter / reddit / parallel (défaut)
            "max_rounds": 100,                     // Facultatif : Nombre maximum de tours de simulation, défaut illimité
            "enable_graph_memory_update": false,   // Facultatif : Activer ou non les mises à jour de mémoire du graphe de connaissances pour les agents
            "force": false                         // Facultatif : Redémarrage forcé (arrêter la simulation en cours et nettoyer les fichiers d'exécution)
        }

    À propos du paramètre force :
        - Après activation, si la simulation est en cours ou terminée, nettoyer les journaux d'exécution
        - Le nettoyage inclut : run_state.json, actions.jsonl, simulation.log, etc.
        - Ne nettoiera pas les fichiers de configuration (simulation_config.json) ni les fichiers de profils
        - Pour les scénarios nécessitant de réexécuter la simulation

    À propos de enable_graph_memory_update :
        - Après activation, tous les agents de la simulation mettront à jour le graphe de connaissances avec leurs actions (posts, commentaires, suivis, etc.)
        - Cela permet au graphe de connaissances de "se souvenir" de la simulation, améliorant la compréhension du contexte et la prise de décision de l'IA
        - Nécessite que le projet associé ait un graph_id valide
        - Utilise un mécanisme de mise à jour par lots pour réduire la surcharge API

    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // Mise à jour mémoire graphe de connaissances activée ou non
                "force_restarted": true               // Redémarrage forcé ou non
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400

        platform = data.get('platform', 'parallel')
        max_rounds = data.get('max_rounds')  # Facultatif : Nombre maximum de tours de simulation
        enable_graph_memory_update = data.get('enable_graph_memory_update', False)  # Facultatif : Activer ou non la mise à jour mémoire du graphe de connaissances
        force = data.get('force', False)  # Facultatif : Redémarrage forcé

        # Vérifier le paramètre max_rounds
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify({
                        "success": False,
                        "error": "max_rounds doit être un entier positif"
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": "max_rounds doit être un entier valide"
                }), 400

        if platform not in ['twitter', 'reddit', 'parallel']:
            return jsonify({
                "success": False,
                "error": f"Type de plateforme invalide : {platform}, options : twitter/reddit/parallel"
            }), 400

        # Vérifier si la simulation est prête
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({
                "success": False,
                "error": f"La simulation n'existe pas : {simulation_id}"
            }), 404

        force_restarted = False
        
        # Gestion intelligente du statut : si la préparation est terminée, réinitialiser le statut à ready
        if state.status != SimulationStatus.READY:
            # Vérifier si la préparation est terminée
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)

            if is_prepared:
                # Préparation terminée, vérifier si la simulation n'est pas déjà en cours
                if state.status == SimulationStatus.RUNNING:
                    # Vérifier si le processus de simulation est réellement en cours
                    run_state = SimulationRunner.get_run_state(simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        # Le processus est effectivement en cours
                        if force:
                            # Mode forcé : arrêter la simulation en cours
                            logger.info(f"Mode forcé : arrêt de la simulation en cours {simulation_id}")
                            try:
                                SimulationRunner.stop_simulation(simulation_id)
                            except Exception as e:
                                logger.warning(f"Avertissement lors de l'arrêt de la simulation : {str(e)}")
                        else:
                            return jsonify({
                                "success": False,
                                "error": f"La simulation est en cours. Veuillez d'abord appeler /stop ou utiliser force=true pour forcer le redémarrage."
                            }), 400

                # Si mode forcé, nettoyer les journaux d'exécution
                if force:
                    logger.info(f"Mode forcé : nettoyage des fichiers d'exécution de simulation pour {simulation_id}")
                    cleanup_result = SimulationRunner.cleanup_simulation_logs(simulation_id)
                    if not cleanup_result.get("success"):
                        logger.warning(f"Avertissement lors du nettoyage des journaux : {cleanup_result.get('errors')}")
                    force_restarted = True

                # Le processus n'existe pas ou est terminé, réinitialiser le statut à ready
                logger.info(f"Simulation {simulation_id} préparation terminée, réinitialisation du statut à ready (statut précédent : {state.status.value})")
                state.status = SimulationStatus.READY
                manager._save_simulation_state(state)
            else:
                # Préparation non terminée
                return jsonify({
                    "success": False,
                    "error": f"Simulation non prête. Statut actuel : {state.status.value}. Veuillez d'abord appeler /prepare"
                }), 400
        
        # Obtenir l'ID du graphe de connaissances (pour la mise à jour mémoire du graphe de connaissances)
        graph_id = None
        if enable_graph_memory_update:
            # Obtenir depuis le statut de simulation ou le graph_id du projet
            graph_id = state.graph_id
            if not graph_id:
                # Essayer d'obtenir depuis le projet
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id
            
            if not graph_id:
                return jsonify({
                    "success": False,
                    "error": "L'activation de la mise à jour mémoire du graphe de connaissances nécessite un graph_id valide, veuillez vous assurer que le graphe du projet est construit"
                }), 400
            
            logger.info(f"Activation de la mise à jour mémoire du graphe de connaissances : simulation_id={simulation_id}, graph_id={graph_id}")
        
        # Démarrer la simulation
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id
        )
        
        # Mettre à jour le statut de la simulation
        state.status = SimulationStatus.RUNNING
        manager._save_simulation_state(state)
        
        response_data = run_state.to_dict()
        if max_rounds:
            response_data['max_rounds_applied'] = max_rounds
        response_data['graph_memory_update_enabled'] = enable_graph_memory_update
        response_data['force_restarted'] = force_restarted
        if enable_graph_memory_update:
            response_data['graph_id'] = graph_id
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Échec du démarrage de la simulation : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/stop', methods=['POST'])
def stop_simulation():
    """
    Arrêter la simulation
    
    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx"  // Requis, ID de simulation
        }
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400
        
        run_state = SimulationRunner.stop_simulation(simulation_id)
        
        # Mettre à jour le statut de la simulation
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Échec de l'arrêt de la simulation : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de surveillance du statut en temps réel ==============

@simulation_bp.route('/<simulation_id>/run-status', methods=['GET'])
def get_run_status(simulation_id: str):
    """
    Obtenir le statut d'exécution en temps réel de la simulation (pour le polling du frontend)
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                "total_rounds": 144,
                "progress_percent": 3.5,
                "simulated_hours": 2,
                "total_simulation_hours": 72,
                "twitter_running": true,
                "reddit_running": true,
                "twitter_actions_count": 150,
                "reddit_actions_count": 200,
                "total_actions_count": 350,
                "started_at": "2025-12-01T10:00:00",
                "updated_at": "2025-12-01T10:30:00"
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "current_round": 0,
                    "total_rounds": 0,
                    "progress_percent": 0,
                    "twitter_actions_count": 0,
                    "reddit_actions_count": 0,
                    "total_actions_count": 0,
                }
            })
        
        return jsonify({
            "success": True,
            "data": run_state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention du statut d'exécution : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/run-status/detail', methods=['GET'])
def get_run_status_detail(simulation_id: str):
    """
    Obtenir le statut d'exécution détaillé de la simulation (incluant toutes les actions)
    
    Pour l'affichage en temps réel par le frontend
    
    Paramètres de requête :
        platform : Filtrer par plateforme (twitter/reddit, facultatif)
    
    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                ...
                "all_actions": [
                    {
                        "round_num": 5,
                        "timestamp": "2025-12-01T10:30:00",
                        "platform": "twitter",
                        "agent_id": 3,
                        "agent_name": "Nom de l'agent",
                        "action_type": "CREATE_POST",
                        "action_args": {"content": "..."},
                        "result": null,
                        "success": true
                    },
                    ...
                ],
                "twitter_actions": [...],  # Toutes les actions de la plateforme Twitter
                "reddit_actions": [...]    # Toutes les actions de la plateforme Reddit
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        platform_filter = request.args.get('platform')
        
        if not run_state:
            return jsonify({
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "runner_status": "idle",
                    "all_actions": [],
                    "twitter_actions": [],
                    "reddit_actions": []
                }
            })
        
        # Obtenir la liste complète des actions
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter
        )
        
        # Obtenir les actions par plateforme
        twitter_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="twitter"
        ) if not platform_filter or platform_filter == "twitter" else []
        
        reddit_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform="reddit"
        ) if not platform_filter or platform_filter == "reddit" else []
        
        # Obtenir les actions du tour actuel (recent_actions montre uniquement le dernier tour)
        current_round = run_state.current_round
        recent_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id,
            platform=platform_filter,
            round_num=current_round
        ) if current_round > 0 else []
        
        # Obtenir les informations de statut de base
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions montre uniquement le contenu du dernier tour des deux plateformes
        result["recent_actions"] = [a.to_dict() for a in recent_actions]
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention du statut détaillé : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/actions', methods=['GET'])
def get_simulation_actions(simulation_id: str):
    """
    Obtenir l'historique des actions des agents de la simulation
    
    Paramètres de requête :
        limit : Nombre de retours (défaut 100)
        offset : Décalage (défaut 0)
        platform : Filtrer par plateforme (twitter/reddit)
        agent_id : Filtrer par ID d'agent
        round_num : Filtrer par tour
    
    Retourne :
        {
            "success": true,
            "data": {
                "count": 100,
                "actions": [...]
            }
        }
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        platform = request.args.get('platform')
        agent_id = request.args.get('agent_id', type=int)
        round_num = request.args.get('round_num', type=int)
        
        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num
        )
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(actions),
                "actions": [a.to_dict() for a in actions]
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention de l'historique des actions : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/timeline', methods=['GET'])
def get_simulation_timeline(simulation_id: str):
    """
    Obtenir la chronologie de la simulation (résumé par tour)
    
    Pour l'affichage de la barre de progression et de la vue chronologique par le frontend
    
    Paramètres de requête :
        start_round : Tour de début (défaut 0)
        end_round : Tour de fin (défaut tous)
    
    Retourne les informations résumées par tour
    """
    try:
        start_round = request.args.get('start_round', 0, type=int)
        end_round = request.args.get('end_round', type=int)
        
        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id,
            start_round=start_round,
            end_round=end_round
        )
        
        return jsonify({
            "success": True,
            "data": {
                "rounds_count": len(timeline),
                "timeline": timeline
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention de la chronologie : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/agent-stats', methods=['GET'])
def get_agent_stats(simulation_id: str):
    """
    Obtenir les statistiques de chaque agent
    
    Pour l'affichage par le frontend du classement d'activité des agents et des statistiques.
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)
        
        return jsonify({
            "success": True,
            "data": {
                "agents_count": len(stats),
                "stats": stats
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des statistiques des agents : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de requête de base de données ==============

@simulation_bp.route('/<simulation_id>/posts', methods=['GET'])
def get_simulation_posts(simulation_id: str):
    """
    Obtenir les posts dans la simulation
    
    Paramètres de requête :
        platform : Type de plateforme (twitter/reddit)
        limit : Nombre de retours (défaut 50)
        offset : Décalage
    
    Retourne la liste des posts (lus depuis la base de données SQLite)
    """
    try:
        platform = request.args.get('platform', 'reddit')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )
        
        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "platform": platform,
                    "count": 0,
                    "posts": [],
                    "message": "La base de données n'existe pas, la simulation n'a peut-être pas encore été exécutée"
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM post 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            posts = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]
            
        except sqlite3.OperationalError:
            posts = []
            total = 0
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "platform": platform,
                "total": total,
                "count": len(posts),
                "posts": posts
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des posts : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/<simulation_id>/comments', methods=['GET'])
def get_simulation_comments(simulation_id: str):
    """
    Obtenir les commentaires dans la simulation (Reddit uniquement)
    
    Paramètres de requête :
        post_id : Filtrer par ID de post (facultatif)
        limit : Nombre de retours
        offset : Décalage
    """
    try:
        post_id = request.args.get('post_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sim_dir = os.path.join(
            os.path.dirname(__file__),
            f'../../uploads/simulations/{simulation_id}'
        )
        
        db_path = os.path.join(sim_dir, "reddit_simulation.db")
        
        if not os.path.exists(db_path):
            return jsonify({
                "success": True,
                "data": {
                    "count": 0,
                    "comments": []
                }
            })
        
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            if post_id:
                cursor.execute("""
                    SELECT * FROM comment 
                    WHERE post_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (post_id, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM comment 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            comments = [dict(row) for row in cursor.fetchall()]
            
        except sqlite3.OperationalError:
            comments = []
        
        conn.close()
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(comments),
                "comments": comments
            }
        })
        
    except Exception as e:
        logger.error(f"Échec de l'obtention des commentaires : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface d'entretien ==============

@simulation_bp.route('/interview', methods=['POST'])
def interview_agent():
    """
    Entretien avec un agent individuel

    Note : Cette fonctionnalité nécessite que la simulation soit en cours d'exécution ou terminée (exécutez la simulation et attendez qu'elle progresse).

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",       // Requis, ID de simulation
            "agent_id": 0,                     // Requis, ID de l'agent
            "prompt": "Que pensez-vous de cela ?",  // Requis, question d'entretien
            "platform": "twitter",             // Facultatif, plateforme spécifiée (twitter/reddit)
                                               // Si non spécifié : les deux plateformes dans les simulations double plateforme
            "timeout": 60                      // Facultatif, délai en secondes, défaut 60
        }

    Retour (si plateforme non spécifiée, retourne les résultats des deux plateformes) :
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Que pensez-vous de cela ?",
                "result": {
                    "agent_id": 0,
                    "prompt": "...",
                    "platforms": {
                        "twitter": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit": {"agent_id": 0, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }

    Retour (plateforme spécifiée) :
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Que pensez-vous de cela ?",
                "result": {
                    "agent_id": 0,
                    "response": "Je pense...",
                    "platform": "twitter",
                    "timestamp": "2025-12-08T10:00:00"
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        agent_id = data.get('agent_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Facultatif : twitter/reddit/None
        timeout = data.get('timeout', 60)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400
        
        if agent_id is None:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir agent_id"
            }), 400
        
        if not prompt:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir le prompt (question d'entretien)"
            }), 400
        
        # Vérifier le paramètre platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Le paramètre platform ne peut être que 'twitter' ou 'reddit'"
            }), 400
        
        # Vérifier le statut de l'environnement
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "L'environnement de simulation n'est pas en cours d'exécution ou est fermé. Veuillez vous assurer que la simulation est démarrée et attendre qu'elle progresse."
            }), 400
        
        # Optimiser le prompt, ajouter un préfixe pour éviter que l'agent appelle des outils
        optimized_prompt = optimize_interview_prompt(prompt)
        
        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Délai d'attente de la réponse d'entretien : {str(e)}"
        }), 504
        
    except Exception as e:
        logger.error(f"Échec de l'entretien : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/batch', methods=['POST'])
def interview_agents_batch():
    """
    Entretien par lot de plusieurs agents

    Note : Cette fonctionnalité nécessite que la simulation soit en cours d'exécution ou terminée.

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",       // Requis, ID de simulation
            "interviews": [                    // Requis, liste d'entretiens
                {
                    "agent_id": 0,
                    "prompt": "Que pensez-vous de A ?",
                    "platform": "twitter"      // Facultatif, interviewer cet agent sur la plateforme spécifiée
                },
                {
                    "agent_id": 1,
                    "prompt": "Que pensez-vous de B ?"  // Si platform non spécifié, utiliser la valeur par défaut
                }
            ],
            "platform": "reddit",              // Facultatif, plateforme par défaut (remplacée par la plateforme de chaque élément)
                                               // Si non spécifié : les deux plateformes dans les simulations double plateforme, plateforme unique dans les simulations mono-plateforme
            "timeout": 120                     // Facultatif, délai en secondes, défaut 120
        }

    Retourne :
        {
            "success": true,
            "data": {
                "interviews_count": 2,
                "result": {
                    "interviews_count": 4,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
                        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        interviews = data.get('interviews')
        platform = data.get('platform')  # Facultatif : twitter/reddit/None
        timeout = data.get('timeout', 120)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify({
                "success": False,
                "error": "Veuillez fournir interviews (liste d'entretiens)"
            }), 400

        # Vérifier le paramètre platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Le paramètre platform ne peut être que 'twitter' ou 'reddit'"
            }), 400

        # Vérifier chaque élément d'entretien
        for i, interview in enumerate(interviews):
            if 'agent_id' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"L'élément {i+1} de la liste d'entretiens manque agent_id"
                }), 400
            if 'prompt' not in interview:
                return jsonify({
                    "success": False,
                    "error": f"L'élément {i+1} de la liste d'entretiens manque prompt"
                }), 400
            # Vérifier la plateforme de chaque élément (si présente)
            item_platform = interview.get('platform')
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify({
                    "success": False,
                    "error": f"Élément {i+1} de la liste d'entretiens : platform doit être 'twitter' ou 'reddit'"
                }), 400

        # Vérifier le statut de l'environnement
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "L'environnement de simulation n'est pas en cours d'exécution ou est fermé. Veuillez vous assurer que la simulation est démarrée et attendre qu'elle progresse."
            }), 400

        # Optimiser le prompt de chaque élément d'entretien, ajouter un préfixe pour éviter que l'agent appelle des outils
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview['prompt'] = optimize_interview_prompt(interview.get('prompt', ''))
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Délai d'attente de la réponse d'entretien par lot : {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Échec de l'entretien par lot : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/all', methods=['POST'])
def interview_all_agents():
    """
    Entretien global - interviewer tous les agents avec la même question

    Note : Cette fonctionnalité nécessite que la simulation soit en cours d'exécution ou terminée.

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",            // Requis, ID de simulation
            "prompt": "Quelle est votre vue d'ensemble sur ce sujet ?",  // Requis, question d'entretien (éviter que l'agent utilise des outils)
            "platform": "reddit",                   // Facultatif, plateforme spécifiée (twitter/reddit)
                                                    // Si non spécifié : les deux plateformes dans les simulations double plateforme, plateforme unique dans les simulations mono-plateforme
            "timeout": 180                          // Facultatif, délai en secondes, défaut 180
        }

    Retourne :
        {
            "success": true,
            "data": {
                "interviews_count": 50,
                "result": {
                    "interviews_count": 100,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        ...
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        prompt = data.get('prompt')
        platform = data.get('platform')  # Facultatif : twitter/reddit/None
        timeout = data.get('timeout', 180)

        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400

        if not prompt:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir le prompt (question d'entretien)"
            }), 400

        # Vérifier le paramètre platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify({
                "success": False,
                "error": "Le paramètre platform ne peut être que 'twitter' ou 'reddit'"
            }), 400

        # Vérifier le statut de l'environnement
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify({
                "success": False,
                "error": "L'environnement de simulation n'est pas en cours d'exécution ou est fermé. Veuillez vous assurer que la simulation est démarrée et attendre qu'elle progresse."
            }), 400

        # Optimiser le prompt, ajouter un préfixe pour éviter que l'agent appelle des outils
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout
        )

        return jsonify({
            "success": result.get("success", False),
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except TimeoutError as e:
        return jsonify({
            "success": False,
            "error": f"Délai d'attente de la réponse d'entretien global : {str(e)}"
        }), 504

    except Exception as e:
        logger.error(f"Échec de l'entretien global : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/interview/history', methods=['POST'])
def get_interview_history():
    """
    Obtenir l'historique des entretiens

    Lire tous les enregistrements d'entretiens de la base de données de simulation

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",  // Requis, ID de simulation
            "platform": "reddit",          // Facultatif, type de plateforme (reddit/twitter)
                                           // Si non spécifié, retourner tout l'historique des deux plateformes
            "agent_id": 0,                 // Facultatif, obtenir uniquement l'historique d'entretien de cet agent
            "limit": 100                   // Facultatif, nombre de retours, défaut 100
        }

    Retourne :
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "Je pense...",
                        "prompt": "Que pensez-vous de cela ?",
                        "timestamp": "2025-12-08T10:00:00",
                        "platform": "reddit"
                    },
                    ...
                ]
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        platform = data.get('platform')  # Si non spécifié, retourner l'historique des deux plateformes
        agent_id = data.get('agent_id')
        limit = data.get('limit', 100)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": {
                "count": len(history),
                "history": history
            }
        })

    except Exception as e:
        logger.error(f"Échec de l'obtention de l'historique des entretiens : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/env-status', methods=['POST'])
def get_env_status():
    """
    Obtenir le statut de l'environnement de simulation

    Vérifier si l'environnement de simulation est actif (peut recevoir des requêtes d'entretien).

    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx"  // Requis, ID de simulation
        }

    Retourne :
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "Environnement en cours d'exécution, prêt à recevoir des requêtes d'entretien"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)
        
        # Obtenir des informations de statut plus détaillées
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = "Environnement en cours d'exécution, prêt à recevoir des requêtes d'entretien"
        else:
            message = "Environnement non en cours d'exécution ou fermé"

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "env_alive": env_alive,
                "twitter_available": env_status.get("twitter_available", False),
                "reddit_available": env_status.get("reddit_available", False),
                "message": message
            }
        })

    except Exception as e:
        logger.error(f"Échec de l'obtention du statut de l'environnement : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/close-env', methods=['POST'])
def close_simulation_env():
    """
    Fermer l'environnement de simulation
    
    Envoyer une commande de fermeture d'environnement à la simulation pour quitter proprement et attendre la complétion.

    Note : Ceci est différent de /stop. /stop termine la simulation de manière abrupte.
    Cette interface laisse la simulation fermer proprement l'environnement et quitter.
    
    Requête (JSON) :
        {
            "simulation_id": "sim_xxxx",  // Requis, ID de simulation
            "timeout": 30                  // Facultatif, délai en secondes, défaut 30
        }
    
    Retourne :
        {
            "success": true,
            "data": {
                "message": "Commande de fermeture d'environnement envoyée",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        timeout = data.get('timeout', 30)
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir simulation_id"
            }), 400
        
        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id,
            timeout=timeout
        )
        
        # Mettre à jour le statut de la simulation
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)
        
        return jsonify({
            "success": result.get("success", False),
            "data": result
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Échec de la fermeture de l'environnement : {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
