"""
Routes API des rapports
Fournit les interfaces de génération, récupération et conversation des rapports de simulation
"""

import os
import traceback
import threading
from flask import request, jsonify, send_file, current_app

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..services.graph_tools import GraphToolsService
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.report')


# ============== Interface de génération de rapport ==============

@report_bp.route('/generate', methods=['POST'])
def generate_report():
    try:
        data = request.get_json() or {}
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({"success": False, "error": "Veuillez fournir simulation_id"}), 400

        force_regenerate = data.get('force_regenerate', False)
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": f"La simulation n'existe pas : {simulation_id}"}), 404

        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({"success": True, "data": {
                    "simulation_id": simulation_id,
                    "report_id": existing_report.report_id,
                    "status": "completed",
                    "message": "Le rapport existe déjà",
                    "already_generated": True
                }})

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Le projet n'existe pas : {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "ID de graphe manquant, veuillez vous assurer que le graphe est construit"}), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({"success": False, "error": "Description de l'exigence de simulation manquante"}), 400

        import uuid
        report_id = f"report_{uuid.uuid4().hex[:12]}"

        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={"simulation_id": simulation_id, "graph_id": graph_id, "report_id": report_id}
        )

        # Initialiser graph_tools dans le contexte Flask AVANT de lancer le thread
        # (current_app n'est pas disponible dans les threads d'arrière-plan)
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            return jsonify({"success": False, "error": "GraphStorage non initialisé — vérifiez la connexion Neo4j"}), 500
        graph_tools = GraphToolsService(storage=storage)

        def run_generate():
            try:
                task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=0, message="Initialisation de l'agent de rapport...")
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    graph_tools=graph_tools
                )
                def progress_callback(stage, progress, message):
                    task_manager.update_task(task_id, progress=progress, message=f"[{stage}] {message}")
                report = agent.generate_report(progress_callback=progress_callback, report_id=report_id)
                ReportManager.save_report(report)
                if report.status == ReportStatus.COMPLETED:
                    task_manager.complete_task(task_id, result={"report_id": report.report_id, "simulation_id": simulation_id, "status": "completed"})
                else:
                    task_manager.fail_task(task_id, report.error or "Échec de la génération du rapport")
            except Exception as e:
                logger.error(f"Échec de la génération du rapport : {str(e)}")
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "report_id": report_id,
            "task_id": task_id,
            "status": "generating",
            "message": "Tâche de génération de rapport démarrée. Consultez la progression via /api/report/generate/status",
            "already_generated": False
        }})

    except Exception as e:
        logger.error(f"Échec du démarrage de la tâche de génération de rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/generate/status', methods=['POST'])
def get_generate_status():
    try:
        data = request.get_json() or {}
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')

        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({"success": True, "data": {
                    "simulation_id": simulation_id,
                    "report_id": existing_report.report_id,
                    "status": "completed",
                    "progress": 100,
                    "message": "Rapport généré",
                    "already_completed": True
                }})

        if not task_id:
            return jsonify({"success": False, "error": "Veuillez fournir task_id ou simulation_id"}), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({"success": False, "error": f"La tâche n'existe pas : {task_id}"}), 404

        return jsonify({"success": True, "data": task.to_dict()})

    except Exception as e:
        logger.error(f"Échec de la consultation du statut de la tâche : {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Interface de récupération de rapport ==============

@report_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Le rapport n'existe pas : {report_id}"}), 404
        return jsonify({"success": True, "data": report.to_dict()})
    except Exception as e:
        logger.error(f"Échec de l'obtention du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/by-simulation/<simulation_id>', methods=['GET'])
def get_report_by_simulation(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        if not report:
            return jsonify({"success": False, "error": f"Aucun rapport disponible pour cette simulation : {simulation_id}", "has_report": False}), 404
        return jsonify({"success": True, "data": report.to_dict()})
    except Exception as e:
        logger.error(f"Échec de l'obtention du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/list', methods=['GET'])
def list_reports():
    try:
        simulation_id = request.args.get('simulation_id')
        limit = request.args.get('limit', 50, type=int)
        reports = ReportManager.list_reports(simulation_id=simulation_id, limit=limit)
        return jsonify({"success": True, "data": [r.to_dict() for r in reports], "count": len(reports)})
    except Exception as e:
        logger.error(f"Échec du listage des rapports : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)
        if not report:
            return jsonify({"success": False, "error": f"Le rapport n'existe pas : {report_id}"}), 404

        md_path = ReportManager._get_report_markdown_path(report_id)
        if not os.path.exists(md_path):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name
            return send_file(temp_path, as_attachment=True, download_name=f"{report_id}.md")

        return send_file(md_path, as_attachment=True, download_name=f"{report_id}.md")

    except Exception as e:
        logger.error(f"Échec du téléchargement du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id: str):
    try:
        success = ReportManager.delete_report(report_id)
        if not success:
            return jsonify({"success": False, "error": f"Le rapport n'existe pas : {report_id}"}), 404
        return jsonify({"success": True, "message": f"Rapport supprimé : {report_id}"})
    except Exception as e:
        logger.error(f"Échec de la suppression du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface de conversation avec l'agent de rapport ==============

@report_bp.route('/chat', methods=['POST'])
def chat_with_report_agent():
    try:
        data = request.get_json() or {}
        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])

        if not simulation_id:
            return jsonify({"success": False, "error": "Veuillez fournir simulation_id"}), 400
        if not message:
            return jsonify({"success": False, "error": "Veuillez fournir le message"}), 400

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if not state:
            return jsonify({"success": False, "error": f"La simulation n'existe pas : {simulation_id}"}), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Le projet n'existe pas : {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "ID de graphe manquant"}), 400

        simulation_requirement = project.simulation_requirement or ""

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé — vérifiez la connexion Neo4j")
        graph_tools = GraphToolsService(storage=storage)

        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
            graph_tools=graph_tools
        )

        result = agent.chat(message=message, chat_history=chat_history)
        return jsonify({"success": True, "data": {"response": result, "simulation_id": simulation_id}})

    except Exception as e:
        logger.error(f"Échec de la conversation : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface de progression et récupération de sections de rapport ==============

@report_bp.route('/<report_id>/progress', methods=['GET'])
def get_report_progress(report_id: str):
    try:
        progress = ReportManager.get_progress(report_id)
        if not progress:
            return jsonify({"success": False, "error": f"Le rapport n'existe pas ou les informations de progression ne sont pas disponibles : {report_id}"}), 404
        return jsonify({"success": True, "data": progress})
    except Exception as e:
        logger.error(f"Échec de l'obtention de la progression du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/sections', methods=['GET'])
def get_report_sections(report_id: str):
    try:
        sections = ReportManager.get_generated_sections(report_id)
        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED
        return jsonify({"success": True, "data": {
            "report_id": report_id,
            "sections": sections,
            "total": len(sections),
            "is_complete": is_complete
        }})
    except Exception as e:
        logger.error(f"Échec de l'obtention de la liste des sections : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/section/<int:section_index>', methods=['GET'])
def get_single_section(report_id: str, section_index: int):
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)
        if not os.path.exists(section_path):
            return jsonify({"success": False, "error": f"La section n'existe pas : section_{section_index:02d}.md"}), 404
        with open(section_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"success": True, "data": {"filename": f"section_{section_index:02d}.md", "content": content}})
    except Exception as e:
        logger.error(f"Échec de l'obtention du contenu de la section : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface de vérification du statut du rapport ==============

@report_bp.route('/check/<simulation_id>', methods=['GET'])
def check_report_status(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        has_report = report is not None
        report_status = report.status.value if report and hasattr(report.status, 'value') else (report.status if report else None)
        report_id = report.report_id if report else None
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED
        return jsonify({"success": True, "data": {
            "simulation_id": simulation_id,
            "has_report": has_report,
            "report_id": report_id,
            "report_status": report_status,
            "interview_unlocked": interview_unlocked
        }})
    except Exception as e:
        logger.error(f"Échec de la vérification du statut du rapport : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface du journal de l'agent ==============

@report_bp.route('/<report_id>/agent-log', methods=['GET'])
def get_agent_log(report_id: str):
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)
        return jsonify({"success": True, "data": log_data})
    except Exception as e:
        logger.error(f"Échec de l'obtention du journal de l'agent : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/agent-log/stream', methods=['GET'])
def stream_agent_log(report_id: str):
    try:
        logs = ReportManager.get_agent_log_stream(report_id)
        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})
    except Exception as e:
        logger.error(f"Échec de l'obtention du journal de l'agent : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface du journal de console ==============

@report_bp.route('/<report_id>/console-log', methods=['GET'])
def get_console_log(report_id: str):
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_console_log(report_id, from_line=from_line)
        return jsonify({"success": True, "data": log_data})
    except Exception as e:
        logger.error(f"Échec de l'obtention du journal de console : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/<report_id>/console-log/stream', methods=['GET'])
def stream_console_log(report_id: str):
    try:
        logs = ReportManager.get_console_log_stream(report_id)
        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})
    except Exception as e:
        logger.error(f"Échec de l'obtention du journal de console : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


# ============== Interface d'appel d'outils (pour le débogage) ==============

@report_bp.route('/tools/search', methods=['POST'])
def search_graph_tool():
    try:
        data = request.get_json() or {}
        graph_id = data.get('graph_id')
        query = data.get('query')
        limit = data.get('limit', 10)
        if not graph_id or not query:
            return jsonify({"success": False, "error": "Veuillez fournir graph_id et query"}), 400
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé — vérifiez la connexion Neo4j")
        tools = GraphToolsService(storage=storage)
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)
        return jsonify({"success": True, "data": result.to_dict()})
    except Exception as e:
        logger.error(f"Échec de la recherche dans le graphe : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@report_bp.route('/tools/statistics', methods=['POST'])
def get_graph_statistics_tool():
    try:
        data = request.get_json() or {}
        graph_id = data.get('graph_id')
        if not graph_id:
            return jsonify({"success": False, "error": "Veuillez fournir graph_id"}), 400
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage non initialisé — vérifiez la connexion Neo4j")
        tools = GraphToolsService(storage=storage)
        result = tools.get_graph_statistics(graph_id)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Échec de l'obtention des statistiques du graphe : {str(e)}")
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()}), 500
