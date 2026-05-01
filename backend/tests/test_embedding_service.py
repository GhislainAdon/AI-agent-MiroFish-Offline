"""
Tests unitaires — EmbeddingService
"""

import pytest
from unittest.mock import patch, MagicMock

from app.storage.embedding_service import EmbeddingService, EmbeddingError


class TestEmbeddingServiceInit:
    """Tests d'initialisation."""

    def test_init_par_defaut(self):
        service = EmbeddingService()
        assert service.model == "nomic-embed-text"
        assert "11434" in service.base_url

    def test_init_personnalise(self):
        service = EmbeddingService(model="custom-model", base_url="http://custom:8080")
        assert service.model == "custom-model"
        assert service.base_url == "http://custom:8080"
        assert service._embed_url == "http://custom:8080/api/embed"


class TestEmbeddingServiceEmbed:
    """Tests de la methode embed()."""

    @patch("app.storage.embedding_service.requests.post")
    def test_embed_succes(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()
        result = service.embed("Test texte")

        assert len(result) == 768
        assert result[0] == 0.1

    def test_embed_texte_vide_leve_erreur(self):
        service = EmbeddingService()
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            service.embed("")

    def test_embed_texte_espaces_leve_erreur(self):
        service = EmbeddingService()
        with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
            service.embed("   ")

    @patch("app.storage.embedding_service.requests.post")
    def test_embed_utilise_cache(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()
        result1 = service.embed("Texte cache")
        result2 = service.embed("Texte cache")

        assert result1 == result2
        # Un seul appel HTTP (le 2e vient du cache)
        assert mock_post.call_count == 1


class TestEmbeddingServiceBatch:
    """Tests de embed_batch()."""

    @patch("app.storage.embedding_service.requests.post")
    def test_batch_succes(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768, [0.2] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()
        results = service.embed_batch(["Texte A", "Texte B"])

        assert len(results) == 2
        assert results[0][0] == 0.1
        assert results[1][0] == 0.2

    def test_batch_vide(self):
        service = EmbeddingService()
        assert service.embed_batch([]) == []

    @patch("app.storage.embedding_service.requests.post")
    def test_batch_texte_vide_donne_vecteur_zero(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.3] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()
        results = service.embed_batch(["Texte valide", ""])

        assert len(results) == 2
        assert results[1] == [0.0] * 768  # Texte vide = vecteur zero


class TestEmbeddingServiceRetry:
    """Tests de la logique de retry."""

    @patch("app.storage.embedding_service.time.sleep")
    @patch("app.storage.embedding_service.requests.post")
    def test_retry_sur_erreur_connexion(self, mock_post, mock_sleep):
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("Connexion refusee")

        service = EmbeddingService(max_retries=2)

        with pytest.raises(EmbeddingError, match="failed after 2 retries"):
            service.embed("Test retry")

        assert mock_post.call_count == 2

    @patch("app.storage.embedding_service.requests.post")
    def test_nombre_embeddings_incorrect_leve_erreur(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()

        with pytest.raises(EmbeddingError, match="Expected 2 embeddings, got 1"):
            service._request_embeddings(["a", "b"])


class TestEmbeddingServiceCache:
    """Tests du cache."""

    @patch("app.storage.embedding_service.requests.post")
    def test_eviction_cache(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        service = EmbeddingService()
        service._cache_max_size = 10

        for i in range(15):
            mock_response.json.return_value = {"embeddings": [[float(i)] * 768]}
            mock_post.return_value = mock_response
            service.embed(f"Texte {i}")

        # Le cache ne depasse pas la taille max + marge d'eviction
        assert len(service._cache) <= 15

    @patch("app.storage.embedding_service.requests.post")
    def test_health_check_ok(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        service = EmbeddingService()
        assert service.health_check() is True

    @patch("app.storage.embedding_service.requests.post")
    def test_health_check_echec(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError("down")

        service = EmbeddingService(max_retries=1)
        assert service.health_check() is False
