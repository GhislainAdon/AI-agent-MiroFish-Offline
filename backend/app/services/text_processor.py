"""
Service de traitement de texte
"""

from typing import List, Optional
from ..utils.file_parser import FileParser, split_text_into_chunks


class TextProcessor:
    """Processeur de texte"""
    
    @staticmethod
    def extract_from_files(file_paths: List[str]) -> str:
        """Extraire le texte de plusieurs fichiers"""
        return FileParser.extract_from_multiple(file_paths)
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Diviser le texte
        
        Args:
            text: Texte original
            chunk_size: Taille des morceaux
            overlap: Taille du chevauchement
            
        Returns:
            Liste des morceaux de texte
        """
        return split_text_into_chunks(text, chunk_size, overlap)
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Prétraiter le texte
        - Supprimer les espaces blancs excédentaires
        - Normaliser les retours à la ligne et les conversions
        
        Args:
            text: Texte source
            
        Returns:
            Texte prétraité
        """
        import re
        
        # Normaliser les retours à la ligne
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Supprimer les lignes vides consécutives (conserver au maximum deux retours à la ligne)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Supprimer les espaces au début et à la fin de chaque ligne
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def get_text_stats(text: str) -> dict:
        """Obtenir les statistiques du texte"""
        return {
            "total_chars": len(text),
            "total_lines": text.count('\n') + 1,
            "total_words": len(text.split()),
        }
