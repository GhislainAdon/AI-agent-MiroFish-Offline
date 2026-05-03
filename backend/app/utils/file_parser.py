"""
Utilitaire d'analyse de fichiers
Prend en charge l'extraction de texte à partir de fichiers PDF, Markdown et TXT
"""

import os
from pathlib import Path
from typing import List, Optional


def _read_text_with_fallback(file_path: str) -> str:
    """
    Lire un fichier texte avec détection automatique de l'encodage si UTF-8 échoue.

    Utilise une stratégie de repli à plusieurs niveaux :
    1. D'abord essayer le décodage UTF-8
    2. Utiliser charset_normalizer pour la détection d'encodage
    3. Repliquer vers chardet pour la détection d'encodage
    4. Enfin utiliser UTF-8 + errors='replace' comme solution de repli

    Args:
        file_path: Chemin du fichier

    Returns:
        Contenu texte décodé
    """
    data = Path(file_path).read_bytes()
    
    # D'abord essayer UTF-8
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass

    # Essayer charset_normalizer pour la détection d'encodage
    encoding = None
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(data).best()
        if best and best.encoding:
            encoding = best.encoding
    except Exception:
        pass

    # Repliquer vers chardet
    if not encoding:
        try:
            import chardet
            result = chardet.detect(data)
            encoding = result.get('encoding') if result else None
        except Exception:
            pass

    # Solution de repli finale : utiliser UTF-8 + replace
    if not encoding:
        encoding = 'utf-8'

    return data.decode(encoding, errors='replace')


class FileParser:
    """Analyseur de fichiers"""

    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.txt'}

    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """
        Extraire le texte d'un fichier

        Args:
            file_path: Chemin du fichier

        Returns:
            Contenu texte extrait
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Le fichier n'existe pas : {file_path}")

        suffix = path.suffix.lower()

        if suffix not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Format de fichier non pris en charge : {suffix}")

        if suffix == '.pdf':
            return cls._extract_from_pdf(file_path)
        elif suffix in {'.md', '.markdown'}:
            return cls._extract_from_md(file_path)
        elif suffix == '.txt':
            return cls._extract_from_txt(file_path)

        raise ValueError(f"Impossible de traiter le format de fichier : {suffix}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extraire le texte d'un PDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF requis : pip install PyMuPDF")

        text_parts = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)

        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_from_md(file_path: str) -> str:
        """Extraire le texte d'un Markdown avec détection automatique de l'encodage"""
        return _read_text_with_fallback(file_path)

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Extraire le texte d'un TXT avec détection automatique de l'encodage"""
        return _read_text_with_fallback(file_path)

    @classmethod
    def extract_from_multiple(cls, file_paths: List[str]) -> str:
        """
        Extraire le texte de plusieurs fichiers et fusionner

        Args:
            file_paths: Liste des chemins de fichiers

        Returns:
            Texte fusionné
        """
        all_texts = []

        for i, file_path in enumerate(file_paths, 1):
            try:
                text = cls.extract_text(file_path)
                filename = Path(file_path).name
                all_texts.append(f"=== Document {i} : {filename} ===\n{text}")
            except Exception as e:
                all_texts.append(f"=== Document {i} : {file_path} (échec de l'extraction : {str(e)}) ===")

        return "\n\n".join(all_texts)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Diviser le texte en morceaux

    Args:
        text: Texte original
        chunk_size: Caractères par morceau
        overlap: Caractères de chevauchement

    Returns:
        Liste de morceaux de texte
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Essayer de diviser aux limites des phrases
        if end < len(text):
            # Trouver la fin de phrase la plus proche
            for sep in ['。', '！', '？', '.\n', '!\n', '?\n', '\n\n', '. ', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Le morceau suivant commence à la position de chevauchement
        start = end - overlap if end < len(text) else len(text)

    return chunks
