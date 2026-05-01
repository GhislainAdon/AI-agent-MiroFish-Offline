"""
Tests unitaires — TaskManager et Task
"""

import pytest
from datetime import datetime, timedelta

from app.models.task import Task, TaskStatus, TaskManager


@pytest.fixture(autouse=True)
def reset_task_manager():
    """Reinitialise le singleton TaskManager entre chaque test."""
    TaskManager._instance = None
    yield
    TaskManager._instance = None


class TestTaskStatus:
    """Tests de l'enum TaskStatus."""

    def test_valeurs_status(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.PROCESSING == "processing"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"


class TestTask:
    """Tests du dataclass Task."""

    def test_creation_task(self):
        now = datetime.now()
        task = Task(
            task_id="t-001",
            task_type="graph_build",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        assert task.task_id == "t-001"
        assert task.progress == 0
        assert task.result is None
        assert task.error is None

    def test_to_dict(self):
        now = datetime.now()
        task = Task(
            task_id="t-002",
            task_type="simulation",
            status=TaskStatus.PROCESSING,
            created_at=now,
            updated_at=now,
            progress=50,
            message="En cours...",
        )
        d = task.to_dict()
        assert d["task_id"] == "t-002"
        assert d["status"] == "processing"
        assert d["progress"] == 50
        assert d["message"] == "En cours..."

    def test_to_dict_contient_toutes_les_cles(self):
        now = datetime.now()
        task = Task(
            task_id="t-003",
            task_type="test",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        d = task.to_dict()
        expected_keys = {
            "task_id", "task_type", "status", "created_at", "updated_at",
            "progress", "message", "progress_detail", "result", "error", "metadata",
        }
        assert set(d.keys()) == expected_keys


class TestTaskManager:
    """Tests du gestionnaire de taches."""

    def test_singleton(self):
        tm1 = TaskManager()
        tm2 = TaskManager()
        assert tm1 is tm2

    def test_create_task(self):
        tm = TaskManager()
        task_id = tm.create_task("graph_build", metadata={"graph_id": "g-001"})
        assert task_id is not None

        task = tm.get_task(task_id)
        assert task is not None
        assert task.task_type == "graph_build"
        assert task.status == TaskStatus.PENDING
        assert task.metadata["graph_id"] == "g-001"

    def test_get_task_inexistant(self):
        tm = TaskManager()
        assert tm.get_task("inexistant-id") is None

    def test_update_task(self):
        tm = TaskManager()
        task_id = tm.create_task("test")

        tm.update_task(task_id, status=TaskStatus.PROCESSING, progress=30, message="Extraction...")

        task = tm.get_task(task_id)
        assert task.status == TaskStatus.PROCESSING
        assert task.progress == 30
        assert task.message == "Extraction..."

    def test_complete_task(self):
        tm = TaskManager()
        task_id = tm.create_task("test")

        tm.complete_task(task_id, result={"entities": 15})

        task = tm.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 100
        assert task.result == {"entities": 15}

    def test_fail_task(self):
        tm = TaskManager()
        task_id = tm.create_task("test")

        tm.fail_task(task_id, error="Connexion Neo4j refusee")

        task = tm.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error == "Connexion Neo4j refusee"

    def test_list_tasks(self):
        tm = TaskManager()
        tm.create_task("type_a")
        tm.create_task("type_b")
        tm.create_task("type_a")

        all_tasks = tm.list_tasks()
        assert len(all_tasks) == 3

        type_a_tasks = tm.list_tasks(task_type="type_a")
        assert len(type_a_tasks) == 2

    def test_list_tasks_ordre_decroissant(self):
        tm = TaskManager()
        id1 = tm.create_task("test")
        id2 = tm.create_task("test")

        tasks = tm.list_tasks()
        # Le plus recent en premier
        assert tasks[0]["task_id"] == id2
        assert tasks[1]["task_id"] == id1

    def test_cleanup_old_tasks(self):
        tm = TaskManager()
        task_id = tm.create_task("test")
        tm.complete_task(task_id, result={})

        # Simuler une tache ancienne
        task = tm.get_task(task_id)
        task.created_at = datetime.now() - timedelta(hours=48)

        tm.cleanup_old_tasks(max_age_hours=24)
        assert tm.get_task(task_id) is None

    def test_cleanup_ne_supprime_pas_taches_recentes(self):
        tm = TaskManager()
        task_id = tm.create_task("test")
        tm.complete_task(task_id, result={})

        tm.cleanup_old_tasks(max_age_hours=24)
        assert tm.get_task(task_id) is not None

    def test_cleanup_ne_supprime_pas_taches_en_cours(self):
        tm = TaskManager()
        task_id = tm.create_task("test")
        tm.update_task(task_id, status=TaskStatus.PROCESSING)

        task = tm.get_task(task_id)
        task.created_at = datetime.now() - timedelta(hours=48)

        tm.cleanup_old_tasks(max_age_hours=24)
        # Les taches en cours ne sont pas supprimees
        assert tm.get_task(task_id) is not None
