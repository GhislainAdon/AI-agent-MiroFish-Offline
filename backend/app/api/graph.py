"""
Routes API liées aux graphes
Utilise le mécanisme de contexte de projet avec persistance de l'état côté serveur
"""

import os
import traceback
import threading
from flask import request, jsonify, current_app

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus

# Obtenir le logger
logger = get_logger('mirofish.api')


def _get_storage():
    """Récupérer Neo4jStorage depuis les extensions de l'application Flask."""
    storage = current_app.extensions.get('neo4j_storage')
    if not storage:
        raise ValueError("GraphStorage non initialisé — vérifiez la connexion Neo4j")
    return storage


def allowed_file(filename: str) -> bool:
    """Vérifier si l'extension du fichier est autorisée"""
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


# ============== Interface de gestion de projet ==============

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    Obtenir les détails du projet
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Le projet n'existe pas : {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    Lister tous les projets
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    Supprimer le projet
    """
    success = ProjectManager.delete_project(project_id)

    if not success:
        return jsonify({
            "success": False,
            "error": f"Le projet n'existe pas ou la suppression a échoué : {project_id}"
        }), 404

    return jsonify({
        "success": True,
        "message": f"Projet supprimé : {project_id}"
    })


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    Réinitialiser le statut du projet (pour reconstruire le graphe)
    """
    project = ProjectManager.get_project(project_id)

    if not project:
        return jsonify({
            "success": False,
            "error": f"Le projet n'existe pas : {project_id}"
        }), 404

    # Réinitialiser à l'état d'ontologie générée
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED

    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)

    return jsonify({
        "success": True,
        "message": f"Projet réinitialisé : {project_id}",
        "data": project.to_dict()
    })


# ============== Interface 1 : Télécharger des fichiers et générer l'ontologie ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    Interface 1 : Télécharger des fichiers et analyser pour générer la définition d'ontologie

    Méthode de requête : multipart/form-data

    Paramètres :
        files : Fichiers téléchargés (PDF/MD/TXT), plusieurs autorisés
        simulation_requirement : Description de l'exigence de simulation (requis)
        project_name : Nom du projet (facultatif)
        additional_context : Notes supplémentaires (facultatif)

    Réponse :
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {
                    "entity_types": [...],
                    "edge_types": [...],
                    "analysis_summary": "..."
                },
                "files": [...],
                "total_text_length": 12345
            }
        }
    """
    try:
        logger.info("=== Démarrage de la génération d'ontologie ===")

        # Obtenir les paramètres
        simulation_requirement = request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Projet sans nom')
        additional_context = request.form.get('additional_context', '')

        logger.debug(f"Nom du projet : {project_name}")
        logger.debug(f"Exigence de simulation : {simulation_requirement[:100]}...")

        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir la description de l'exigence de simulation (simulation_requirement)"
            }), 400

        # Obtenir les fichiers téléchargés
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({
                "success": False,
                "error": "Veuillez télécharger au moins un fichier document"
            }), 400

        # Créer le projet
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = simulation_requirement
        logger.info(f"Projet créé : {project.project_id}")
        
        # Sauvegarder les fichiers et extraire le texte
        document_texts = []
        all_text = ""

        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # Sauvegarder le fichier dans le répertoire du projet
                file_info = ProjectManager.save_file_to_project(
                    project.project_id,
                    file,
                    file.filename
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"]
                })

                # Extraire le texte
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                document_texts.append(text)
                all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"

        if not document_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": "Aucun document n'a été traité avec succès. Veuillez vérifier le format du fichier"
            }), 400

        # Sauvegarder le texte extrait
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        logger.info(f"Extraction de texte terminée, total {len(all_text)} caractères")

        # Générer l'ontologie
        logger.info("Appel au LLM pour générer la définition d'ontologie...")
        generator = OntologyGenerator()
        ontology = generator.generate(
            document_texts=document_texts,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context if additional_context else None
        )

        # Sauvegarder l'ontologie dans le projet
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(f"Génération d'ontologie terminée : {entity_count} types d'entités, {edge_count} types de relations")
        
        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", [])
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        ProjectManager.save_project(project)
        logger.info(f"=== Génération d'ontologie terminée === ID du projet : {project.project_id}")
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "project_name": project.name,
                "ontology": project.ontology,
                "analysis_summary": project.analysis_summary,
                "files": project.files,
                "total_text_length": project.total_text_length
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface 2 : Construire le graphe ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """
    Interface 2 : Construire le graphe à partir de project_id

    Requête (JSON) :
        {
            "project_id": "proj_xxxx",  // Requis : depuis l'interface 1
            "graph_name": "Nom du graphe",    // Facultatif
            "chunk_size": 500,          // Facultatif, défaut 500
            "chunk_overlap": 50         // Facultatif, défaut 50
        }

    Réponse :
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "Tâche de construction de graphe démarrée"
            }
        }
    """
    try:
        logger.info("=== Démarrage de la construction du graphe ===")

        # Analyser la requête
        data = request.get_json() or {}
        project_id = data.get('project_id')
        logger.debug(f"Paramètres de la requête : project_id={project_id}")
        
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Veuillez fournir project_id"
            }), 400

        # Obtenir le projet
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"Le projet n'existe pas : {project_id}"
            }), 404

        # Vérifier le statut du projet
        force = data.get('force', False)  # Forcer la reconstruction

        if project.status == ProjectStatus.CREATED:
            return jsonify({
                "success": False,
                "error": "Le projet n'a pas encore généré d'ontologie. Veuillez d'abord appeler /ontology/generate"
            }), 400

        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify({
                "success": False,
                "error": "Le graphe est en cours de construction. Ne soumettez pas plusieurs fois. Pour forcer la reconstruction, ajoutez force: true",
                "task_id": project.graph_build_task_id
            }), 400

        # Si reconstruction forcée, réinitialiser le statut
        if force and project.status in [ProjectStatus.GRAPH_BUILDING, ProjectStatus.FAILED, ProjectStatus.GRAPH_COMPLETED]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            project.error = None

        # Obtenir la configuration
        graph_name = data.get('graph_name', project.name or 'Graphe MiroFish')
        chunk_size = data.get('chunk_size', project.chunk_size or Config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = data.get('chunk_overlap', project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP)

        # Mettre à jour la configuration du projet
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap

        # Obtenir le texte extrait
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify({
                "success": False,
                "error": "Texte extrait introuvable"
            }), 400

        # Obtenir l'ontologie
        ontology = project.ontology
        if not ontology:
            return jsonify({
                "success": False,
                "error": "Définition d'ontologie introuvable"
            }), 400

        # Obtenir le stockage dans le contexte de la requête (le thread d'arrière-plan ne peut pas accéder à current_app)
        storage = _get_storage()

        # Créer une tâche asynchrone
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Construire le graphe : {graph_name}")
        logger.info(f"Tâche de construction de graphe créée : task_id={task_id}, project_id={project_id}")
        
        # Mettre à jour le statut du projet
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        ProjectManager.save_project(project)

        # Démarrer la tâche en arrière-plan
        def build_task():
            build_logger = get_logger('mirofish.build')
            try:
                build_logger.info(f"[{task_id}] Démarrage de la construction du graphe...")
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    message="Initialisation du service de construction de graphe..."
                )

                # Créer le service de construction de graphe (stockage passé depuis la fermeture externe)
                builder = GraphBuilderService(storage=storage)

                # Découper le texte
                task_manager.update_task(
                    task_id,
                    message="Découpage du texte...",
                    progress=5
                )
                chunks = TextProcessor.split_text(
                    text,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                total_chunks = len(chunks)

                # Créer le graphe
                task_manager.update_task(
                    task_id,
                    message="Création du graphe Zep...",
                    progress=10
                )
                graph_id = builder.create_graph(name=graph_name)

                # Mettre à jour le graph_id du projet
                project.graph_id = graph_id
                ProjectManager.save_project(project)

                # Définir l'ontologie
                task_manager.update_task(
                    task_id,
                    message="Définition de l'ontologie...",
                    progress=15
                )
                builder.set_ontology(graph_id, ontology)
                
                # Ajouter le texte (la signature de progress_callback est (msg, progress_ratio))
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )

                task_manager.update_task(
                    task_id,
                    message=f"Début de l'ajout de {total_chunks} morceaux de texte...",
                    progress=15
                )

                episode_uuids = builder.add_text_batches(
                    graph_id,
                    chunks,
                    batch_size=3,
                    progress_callback=add_progress_callback
                )

                # Le traitement Neo4j est synchrone, pas besoin d'attendre
                task_manager.update_task(
                    task_id,
                    message="Traitement du texte terminé, génération des données du graphe...",
                    progress=90
                )

                # Obtenir les données du graphe
                task_manager.update_task(
                    task_id,
                    message="Récupération des données du graphe...",
                    progress=95
                )
                graph_data = builder.get_graph_data(graph_id)

                # Mettre à jour le statut du projet
                project.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(project)

                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                build_logger.info(f"[{task_id}] Construction du graphe terminée : graph_id={graph_id}, nœuds={node_count}, arêtes={edge_count}")

                # Terminé
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="Construction du graphe terminée",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks
                    }
                )

            except Exception as e:
                # Mettre à jour le statut du projet à échoué
                build_logger.error(f"[{task_id}] Échec de la construction du graphe : {str(e)}")
                build_logger.debug(traceback.format_exc())

                project.status = ProjectStatus.FAILED
                project.error = str(e)
                ProjectManager.save_project(project)

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Échec de la construction : {str(e)}",
                    error=traceback.format_exc()
                )

        # Démarrer le thread d'arrière-plan
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": "Tâche de construction de graphe démarrée. Consultez la progression via /task/{task_id}"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Interface de consultation des tâches ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Consulter le statut de la tâche
    """
    task = TaskManager().get_task(task_id)

    if not task:
        return jsonify({
            "success": False,
            "error": f"La tâche n'existe pas : {task_id}"
        }), 404

    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    Lister toutes les tâches
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== Interface des données du graphe ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    Obtenir les données du graphe (nœuds et arêtes)
    """
    try:
        storage = _get_storage()
        builder = GraphBuilderService(storage=storage)
        graph_data = builder.get_graph_data(graph_id)

        return jsonify({
            "success": True,
            "data": graph_data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    Supprimer le graphe
    """
    try:
        storage = _get_storage()
        builder = GraphBuilderService(storage=storage)
        builder.delete_graph(graph_id)

        return jsonify({
            "success": True,
            "message": f"Graphe supprimé : {graph_id}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
