"""
Fixtures partagees pour les tests MiroFish-Offline.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# Forcer les variables d'environnement AVANT l'import de Config
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL_NAME", "llama3.2:1b")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "mirofish")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("EMBEDDING_BASE_URL", "http://localhost:11434")


@pytest.fixture
def mock_openai_client():
    """Mock du client OpenAI pour les tests LLM."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"result": "ok"}'
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_storage():
    """Mock du GraphStorage pour les tests sans Neo4j."""
    storage = MagicMock()
    storage.get_all_nodes.return_value = []
    storage.get_all_edges.return_value = []
    storage.get_node.return_value = None
    storage.get_node_edges.return_value = []
    return storage


@pytest.fixture
def sample_nodes():
    """Noeuds exemple pour les tests d'entites."""
    return [
        {
            "uuid": "node-001",
            "name": "Sophie Martin",
            "labels": ["Entity", "Person"],
            "summary": "PDG de TechWave SAS",
            "attributes": {"role": "CEO"},
        },
        {
            "uuid": "node-002",
            "name": "TechWave SAS",
            "labels": ["Entity", "Organization"],
            "summary": "Startup technologique basee a Paris",
            "attributes": {"sector": "tech"},
        },
        {
            "uuid": "node-003",
            "name": "CNIL",
            "labels": ["Entity", "GovernmentAgency"],
            "summary": "Commission Nationale de l'Informatique et des Libertes",
            "attributes": {},
        },
        {
            "uuid": "node-004",
            "name": "Noeud generique",
            "labels": ["Entity"],
            "summary": "Un noeud sans type specifique",
            "attributes": {},
        },
    ]


@pytest.fixture
def sample_edges():
    """Aretes exemple pour les tests."""
    return [
        {
            "uuid": "edge-001",
            "name": "DIRIGE",
            "source_node_uuid": "node-001",
            "target_node_uuid": "node-002",
            "fact": "Sophie Martin dirige TechWave SAS",
        },
        {
            "uuid": "edge-002",
            "name": "REGULE",
            "source_node_uuid": "node-003",
            "target_node_uuid": "node-002",
            "fact": "La CNIL regule TechWave SAS",
        },
    ]


@pytest.fixture
def flask_app():
    """Application Flask de test avec Neo4j mocke."""
    with patch("app.storage.Neo4jStorage") as mock_neo4j_cls:
        mock_neo4j_cls.return_value = MagicMock()
        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        yield app


@pytest.fixture
def client(flask_app):
    """Client de test Flask."""
    return flask_app.test_client()
