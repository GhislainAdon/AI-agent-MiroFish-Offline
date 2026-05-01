"""
Tests unitaires — EntityReader et EntityNode
"""

import pytest

from app.services.entity_reader import EntityReader, EntityNode, FilteredEntities


class TestEntityNode:
    """Tests du dataclass EntityNode."""

    def test_creation(self):
        node = EntityNode(
            uuid="n-001",
            name="Sophie Martin",
            labels=["Entity", "Person"],
            summary="PDG",
            attributes={"role": "CEO"},
        )
        assert node.uuid == "n-001"
        assert node.related_edges == []
        assert node.related_nodes == []

    def test_get_entity_type_person(self):
        node = EntityNode(
            uuid="n-001", name="Test", labels=["Entity", "Person"],
            summary="", attributes={},
        )
        assert node.get_entity_type() == "Person"

    def test_get_entity_type_avec_node_label(self):
        node = EntityNode(
            uuid="n-001", name="Test", labels=["Node", "Entity", "Organization"],
            summary="", attributes={},
        )
        assert node.get_entity_type() == "Organization"

    def test_get_entity_type_sans_type_specifique(self):
        node = EntityNode(
            uuid="n-001", name="Test", labels=["Entity"],
            summary="", attributes={},
        )
        assert node.get_entity_type() is None

    def test_to_dict(self):
        node = EntityNode(
            uuid="n-001", name="Test", labels=["Entity", "Person"],
            summary="desc", attributes={"a": 1},
        )
        d = node.to_dict()
        assert d["uuid"] == "n-001"
        assert d["name"] == "Test"
        assert d["labels"] == ["Entity", "Person"]
        assert d["related_edges"] == []


class TestEntityReaderFilter:
    """Tests du filtrage d'entites."""

    def test_filtre_entites_avec_type(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001")

        # node-004 (labels=["Entity"]) devrait etre exclu
        assert result.filtered_count == 3
        assert result.total_count == 4
        assert "Person" in result.entity_types
        assert "Organization" in result.entity_types
        assert "GovernmentAgency" in result.entity_types

    def test_filtre_par_type_specifique(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001", defined_entity_types=["Person"])

        assert result.filtered_count == 1
        assert result.entities[0].name == "Sophie Martin"

    def test_filtre_graphe_vide(self, mock_storage):
        mock_storage.get_all_nodes.return_value = []
        mock_storage.get_all_edges.return_value = []

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001")

        assert result.filtered_count == 0
        assert result.total_count == 0
        assert len(result.entity_types) == 0

    def test_enrichissement_aretes(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001", enrich_with_edges=True)

        # Sophie Martin (node-001) a une arete sortante vers TechWave (node-002)
        sophie = [e for e in result.entities if e.name == "Sophie Martin"][0]
        assert len(sophie.related_edges) == 1
        assert sophie.related_edges[0]["direction"] == "outgoing"
        assert sophie.related_edges[0]["edge_name"] == "DIRIGE"

        # TechWave (node-002) a 2 aretes entrantes
        techwave = [e for e in result.entities if e.name == "TechWave SAS"][0]
        assert len(techwave.related_edges) == 2

    def test_sans_enrichissement_aretes(self, mock_storage, sample_nodes):
        mock_storage.get_all_nodes.return_value = sample_nodes

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001", enrich_with_edges=False)

        for entity in result.entities:
            assert entity.related_edges == []
            assert entity.related_nodes == []

        # get_all_edges ne devrait pas etre appele
        mock_storage.get_all_edges.assert_not_called()

    def test_noeuds_relies_inclus(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        result = reader.filter_defined_entities("graph-001")

        sophie = [e for e in result.entities if e.name == "Sophie Martin"][0]
        assert len(sophie.related_nodes) == 1
        assert sophie.related_nodes[0]["name"] == "TechWave SAS"


class TestEntityReaderGetByType:
    """Tests de la recuperation par type."""

    def test_get_entities_by_type(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        orgs = reader.get_entities_by_type("graph-001", "Organization")

        assert len(orgs) == 1
        assert orgs[0].name == "TechWave SAS"

    def test_get_entities_type_inexistant(self, mock_storage, sample_nodes, sample_edges):
        mock_storage.get_all_nodes.return_value = sample_nodes
        mock_storage.get_all_edges.return_value = sample_edges

        reader = EntityReader(mock_storage)
        result = reader.get_entities_by_type("graph-001", "TypeInexistant")

        assert len(result) == 0


class TestEntityReaderGetWithContext:
    """Tests de get_entity_with_context."""

    def test_entite_trouvee(self, mock_storage):
        mock_storage.get_node.return_value = {
            "uuid": "n-001", "name": "Sophie", "labels": ["Entity", "Person"],
            "summary": "PDG", "attributes": {},
        }
        mock_storage.get_node_edges.return_value = [
            {
                "name": "DIRIGE",
                "source_node_uuid": "n-001",
                "target_node_uuid": "n-002",
                "fact": "Sophie dirige TechWave",
            }
        ]

        reader = EntityReader(mock_storage)
        entity = reader.get_entity_with_context("graph-001", "n-001")

        assert entity is not None
        assert entity.name == "Sophie"
        assert len(entity.related_edges) == 1

    def test_entite_non_trouvee(self, mock_storage):
        mock_storage.get_node.return_value = None

        reader = EntityReader(mock_storage)
        entity = reader.get_entity_with_context("graph-001", "inexistant")

        assert entity is None
