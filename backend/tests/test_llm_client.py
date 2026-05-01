"""
Tests unitaires — LLMClient
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.utils.llm_client import LLMClient


class TestLLMClientInit:
    """Tests d'initialisation du client LLM."""

    @patch("app.utils.llm_client.OpenAI")
    def test_init_avec_parametres(self, mock_openai):
        client = LLMClient(api_key="key", base_url="http://test:1234/v1", model="test-model")
        assert client.api_key == "key"
        assert client.base_url == "http://test:1234/v1"
        assert client.model == "test-model"

    @patch("app.utils.llm_client.OpenAI")
    def test_init_sans_api_key_leve_erreur(self, mock_openai):
        with patch("app.utils.llm_client.Config") as mock_config:
            mock_config.LLM_API_KEY = None
            mock_config.LLM_BASE_URL = "http://localhost:11434/v1"
            mock_config.LLM_MODEL_NAME = "test"
            with pytest.raises(ValueError, match="LLM_API_KEY not configured"):
                LLMClient(api_key=None)

    @patch("app.utils.llm_client.OpenAI")
    def test_detection_ollama(self, mock_openai):
        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        assert client._is_ollama() is True

    @patch("app.utils.llm_client.OpenAI")
    def test_detection_non_ollama(self, mock_openai):
        client = LLMClient(api_key="k", base_url="http://api.openai.com/v1", model="m")
        assert client._is_ollama() is False


class TestLLMClientChat:
    """Tests de la methode chat()."""

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_retourne_contenu(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Bonjour le monde"
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == "Bonjour le monde"

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_supprime_balises_think(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "<think>reflexion interne</think>Reponse finale"
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        result = client.chat([{"role": "user", "content": "test"}])

        assert result == "Reponse finale"
        assert "<think>" not in result

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_ollama_envoie_num_ctx(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" in call_kwargs
        assert call_kwargs["extra_body"]["options"]["num_ctx"] == 8192

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_non_ollama_pas_de_num_ctx(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://api.openai.com/v1", model="m")
        client.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "extra_body" not in call_kwargs


class TestLLMClientChatJson:
    """Tests de la methode chat_json()."""

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_json_parse_valide(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "MiroFish", "version": 2}'
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        result = client.chat_json([{"role": "user", "content": "test"}])

        assert result == {"name": "MiroFish", "version": 2}

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_json_nettoie_markdown(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")
        result = client.chat_json([{"role": "user", "content": "test"}])

        assert result == {"key": "value"}

    @patch("app.utils.llm_client.OpenAI")
    def test_chat_json_invalide_leve_erreur(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ceci n'est pas du JSON"
        mock_client.chat.completions.create.return_value = mock_response

        client = LLMClient(api_key="k", base_url="http://localhost:11434/v1", model="m")

        with pytest.raises(ValueError, match="Invalid JSON format"):
            client.chat_json([{"role": "user", "content": "test"}])
