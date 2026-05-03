"""
Client LLM (Wrapper)
Appels API unifiés au format OpenAI
Prend en charge le paramètre Ollama num_ctx pour éviter la troncature du prompt
"""

import json
import os
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """Client LLM"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY non configurée")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Taille de la fenêtre de contexte Ollama — empêche la troncature du prompt.
        # Lu depuis la variable d'environnement OLLAMA_NUM_CTX, par défaut 8192 (la valeur par défaut d'Ollama est seulement 2048).
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Vérifier si nous communiquons avec un serveur Ollama."""
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Envoyer une requête de chat

        Args:
            messages: Liste de messages
            temperature: Paramètre de température
            max_tokens: Nombre maximal de jetons
            response_format: Format de réponse (par ex. mode JSON)

        Returns:
            Texte de réponse du modèle
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Pour Ollama : passer num_ctx via extra_body pour éviter la troncature du prompt
        if self._is_ollama() and self._num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._num_ctx}
            }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Certains modèles (comme MiniMax M2.5) incluent du contenu de réflexion dans la réponse, il faut le supprimer
        content = re.sub(r'<think\>[\s\S]*?\</think\>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Envoyer une requête de chat et retourner du JSON

        Args:
            messages: Liste de messages
            temperature: Paramètre de température
            max_tokens: Nombre maximal de jetons

        Returns:
            Objet JSON analysé
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Nettoyer les marqueurs de bloc de code markdown
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Format JSON invalide provenant du LLM : {cleaned_response}")
