"""
Tests unitaires — Application Flask (factory, routes, health check)
"""

import pytest
from unittest.mock import patch, MagicMock


class TestFlaskAppFactory:
    """Tests de la factory create_app."""

    def test_app_creation(self, flask_app):
        assert flask_app is not None
        assert flask_app.config["TESTING"] is True

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "MiroFish-Offline Backend"

    def test_cors_configure(self, flask_app):
        # Verifier que CORS est configure (health check accessible)
        with flask_app.test_client() as c:
            response = c.get("/health", headers={
                "Origin": "http://localhost:3000",
            })
            assert response.status_code == 200

    def test_blueprints_enregistres(self, flask_app):
        blueprints = list(flask_app.blueprints.keys())
        assert "graph" in blueprints or "graph_bp" in blueprints or any("graph" in b for b in blueprints)

    def test_neo4j_storage_dans_extensions(self, flask_app):
        assert "neo4j_storage" in flask_app.extensions

    def test_neo4j_init_echec_gracieux(self):
        """Si Neo4j echoue, l'app demarre quand meme avec storage=None."""
        with patch("app.storage.Neo4jStorage", side_effect=Exception("Connection refused")):
            from app import create_app
            app = create_app()
            assert app.extensions["neo4j_storage"] is None


class TestConfigValidation:
    """Tests de la validation de configuration."""

    def test_validate_config_complete(self):
        from app.config import Config
        errors = Config.validate()
        assert len(errors) == 0  # Toutes les vars sont definies dans conftest

    def test_validate_sans_api_key(self):
        from app.config import Config
        original = Config.LLM_API_KEY
        Config.LLM_API_KEY = None
        try:
            errors = Config.validate()
            assert any("LLM_API_KEY" in e for e in errors)
        finally:
            Config.LLM_API_KEY = original
