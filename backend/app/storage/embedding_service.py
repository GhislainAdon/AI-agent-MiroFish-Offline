"""
EmbeddingService — incorporation locale via l'API Ollama

Remplace l'incorporation intégrée de Zep Cloud avec le modèle local nomic-embed-text.
Utilise le point de terminaison /api/embed d'Ollama pour la génération de vecteurs (768 dimensions).
"""

import time
import logging
from typing import List, Optional
from functools import lru_cache

import requests

from ..config import Config

logger = logging.getLogger('mirofish.embedding')


class EmbeddingService:
    """Générer des incorporations en utilisant le serveur Ollama local."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.model = model or Config.EMBEDDING_MODEL
        self.base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        self._embed_url = f"{self.base_url}/api/embed"

        # Cache simple en mémoire (texte -> vecteur d'incorporation)
        # Utilisation d'un dictionnaire au lieu de lru_cache car les listes ne sont pas hachables
        self._cache: dict[str, List[float]] = {}
        self._cache_max_size = 2000

    def embed(self, text: str) -> List[float]:
        """
        Générer une incorporation pour un texte unique.

        Args:
            text: Texte en entrée à incorporer

        Returns:
            Vecteur de flottants à 768 dimensions

        Raises:
            EmbeddingError: Si la requête Ollama échoue après les tentatives
        """
        if not text or not text.strip():
            raise EmbeddingError("Impossible d'incorporer un texte vide")

        text = text.strip()

        # Vérifier le cache
        if text in self._cache:
            return self._cache[text]

        vectors = self._request_embeddings([text])
        vector = vectors[0]

        # Mettre en cache le résultat
        self._cache_put(text, vector)

        return vector

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Générer des incorporations pour plusieurs textes.

        Traite par lots pour éviter de surcharger Ollama.

        Args:
            texts: Liste des textes en entrée
            batch_size: Nombre de textes par requête

        Returns:
            Liste de vecteurs d'incorporation (même ordre que l'entrée)
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Vérifier le cache d'abord
        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if text in self._cache:
                results[i] = self._cache[text]
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                # Texte vide — vecteur nul
                results[i] = [0.0] * 768

        # Incorporer par lot les textes non mis en cache
        if uncached_texts:
            all_vectors: List[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                vectors = self._request_embeddings(batch)
                all_vectors.extend(vectors)

            # Placer les résultats et mettre en cache
            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put(text, vec)

        return results  # type: ignore

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Envoyer une requête HTTP au point de terminaison Ollama /api/embed avec nouvelle tentative.

        Args:
            texts: Liste des textes à incorporer (Ollama prend en charge le lot en une seule requête)

        Returns:
            Liste de vecteurs d'incorporation
        """
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self._embed_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"{len(texts)} incorporations attendues, {len(embeddings)} obtenues"
                    )

                return embeddings

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(
                    f"Connexion Ollama échouée (tentative {attempt + 1}/{self.max_retries}) : {e}"
                )
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    f"Délai d'attente de la requête Ollama dépassé (tentative {attempt + 1}/{self.max_retries})"
                )
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.error(f"Erreur HTTP Ollama : {e.response.status_code} - {e.response.text}")
                if e.response.status_code >= 500:
                    # Erreur serveur — retenter
                    pass
                else:
                    # Erreur client (4xx) — ne pas retenter
                    raise EmbeddingError(f"L'incorporation Ollama a échoué : {e}") from e
            except (KeyError, ValueError) as e:
                raise EmbeddingError(f"Réponse Ollama invalide : {e}") from e

            # Intervalle exponentiel
            if attempt < self.max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"Nouvelle tentative dans {wait}s...")
                time.sleep(wait)

        raise EmbeddingError(
            f"L'incorporation Ollama a échoué après {self.max_retries} tentatives : {last_error}"
        )

    def _cache_put(self, text: str, vector: List[float]) -> None:
        """Ajouter au cache, en évictant les entrées les plus anciennes si plein."""
        if len(self._cache) >= self._cache_max_size:
            # Supprimer environ 10 % des entrées les plus anciennes
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[text] = vector

    def health_check(self) -> bool:
        """Vérifier si le point de terminaison d'incorporation Ollama est accessible."""
        try:
            vec = self.embed("vérification de l'état de santé")
            return len(vec) > 0
        except Exception:
            return False


class EmbeddingError(Exception):
    """Levée lorsque la génération d'incorporation échoue."""
    pass
