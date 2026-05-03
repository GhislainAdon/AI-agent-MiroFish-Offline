"""
Gestion des statuts de tâche
Suit les tâches de longue durée (comme la construction de graphes)
"""

import uuid
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Énumération des statuts de tâche"""
    PENDING = "pending"          # En attente
    PROCESSING = "processing"    # En cours de traitement
    COMPLETED = "completed"      # Terminé
    FAILED = "failed"            # Échoué


@dataclass
class Task:
    """Classe de données de tâche"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    progress: int = 0              # Pourcentage de progression global 0-100
    message: str = ""              # Message de statut
    result: Optional[Dict] = None  # Résultat de la tâche
    error: Optional[str] = None    # Message d'erreur
    metadata: Dict = field(default_factory=dict)  # Métadonnées supplémentaires
    progress_detail: Dict = field(default_factory=dict)  # Informations détaillées sur la progression

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Gestionnaire de tâches
    Gestion de l'état des tâches thread-safe
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Modèle singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
        return cls._instance

    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Créer une nouvelle tâche

        Args:
            task_type: Type de tâche
            metadata: Métadonnées supplémentaires

        Returns:
            Identifiant de la tâche
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()

        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

        with self._task_lock:
            self._tasks[task_id] = task

        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtenir une tâche"""
        with self._task_lock:
            return self._tasks.get(task_id)

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """
        Mettre à jour le statut d'une tâche

        Args:
            task_id: Identifiant de la tâche
            status: Nouveau statut
            progress: Progression
            message: Message
            result: Résultat
            error: Message d'erreur
            progress_detail: Informations détaillées sur la progression
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.updated_at = datetime.now()
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if progress_detail is not None:
                    task.progress_detail = progress_detail

    def complete_task(self, task_id: str, result: Dict):
        """Marquer la tâche comme terminée"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Tâche terminée",
            result=result
        )

    def fail_task(self, task_id: str, error: str):
        """Marquer la tâche comme échouée"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message="Tâche échouée",
            error=error
        )

    def list_tasks(self, task_type: Optional[str] = None) -> list:
        """Lister les tâches"""
        with self._task_lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Nettoyer les anciennes tâches"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        with self._task_lock:
            old_ids = [
                tid for tid, task in self._tasks.items()
                if task.created_at < cutoff and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            for tid in old_ids:
                del self._tasks[tid]
