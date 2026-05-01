"""
Tests unitaires — FileParser et split_text_into_chunks
"""

import os
import pytest
import tempfile

from app.utils.file_parser import FileParser, split_text_into_chunks


class TestFileParser:
    """Tests d'extraction de texte."""

    def test_extract_txt(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Bonjour le monde MiroFish", encoding="utf-8")

        result = FileParser.extract_text(str(txt_file))
        assert "Bonjour le monde MiroFish" in result

    def test_extract_md(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Titre\n\nContenu markdown", encoding="utf-8")

        result = FileParser.extract_text(str(md_file))
        assert "# Titre" in result
        assert "Contenu markdown" in result

    def test_fichier_inexistant(self):
        with pytest.raises(FileNotFoundError):
            FileParser.extract_text("/chemin/inexistant/fichier.txt")

    def test_format_non_supporte(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\na,b", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file format"):
            FileParser.extract_text(str(csv_file))

    def test_supported_extensions(self):
        assert ".pdf" in FileParser.SUPPORTED_EXTENSIONS
        assert ".md" in FileParser.SUPPORTED_EXTENSIONS
        assert ".txt" in FileParser.SUPPORTED_EXTENSIONS
        assert ".markdown" in FileParser.SUPPORTED_EXTENSIONS
        assert ".csv" not in FileParser.SUPPORTED_EXTENSIONS

    def test_extract_txt_encodage_latin1(self, tmp_path):
        txt_file = tmp_path / "latin.txt"
        txt_file.write_bytes("Cafe resume etudiante".encode("latin-1"))

        result = FileParser.extract_text(str(txt_file))
        assert "Cafe" in result

    def test_extract_from_multiple(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("Premier document", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("Second document", encoding="utf-8")

        result = FileParser.extract_from_multiple([str(f1), str(f2)])
        assert "Premier document" in result
        assert "Second document" in result
        assert "Document 1" in result
        assert "Document 2" in result

    def test_extract_from_multiple_fichier_invalide(self, tmp_path):
        f1 = tmp_path / "ok.txt"
        f1.write_text("Contenu valide", encoding="utf-8")

        result = FileParser.extract_from_multiple([str(f1), "/inexistant.txt"])
        assert "Contenu valide" in result
        assert "extraction failed" in result


class TestSplitTextIntoChunks:
    """Tests du decoupage de texte."""

    def test_texte_court_un_seul_chunk(self):
        chunks = split_text_into_chunks("Court texte", chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Court texte"

    def test_texte_vide(self):
        chunks = split_text_into_chunks("", chunk_size=500)
        assert chunks == []

    def test_texte_espaces_seulement(self):
        chunks = split_text_into_chunks("   \n  \t  ", chunk_size=500)
        assert chunks == []

    def test_decoupage_long_texte(self):
        texte = "Phrase numero un. " * 100  # ~1900 caracteres
        chunks = split_text_into_chunks(texte, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        # Tous les chunks sont non vides
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap_fonctionne(self):
        texte = "A" * 300
        chunks = split_text_into_chunks(texte, chunk_size=200, overlap=50)
        assert len(chunks) >= 2

    def test_decoupe_aux_limites_de_phrases(self):
        texte = "Premiere phrase longue ici. Deuxieme phrase longue la. Troisieme phrase ici."
        chunks = split_text_into_chunks(texte, chunk_size=40, overlap=5)
        assert len(chunks) >= 2
