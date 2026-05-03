"""
Schéma Neo4j — requêtes Cypher pour la création d'index et la gestion du schéma.

Appelé par Neo4jStorage.create_graph() pour configurer les index vectoriels et plein texte.
"""

# Contraintes
CREATE_GRAPH_UUID_CONSTRAINT = """
CREATE CONSTRAINT graph_uuid IF NOT EXISTS
FOR (g:Graph) REQUIRE g.graph_id IS UNIQUE
"""

CREATE_ENTITY_UUID_CONSTRAINT = """
CREATE CONSTRAINT entity_uuid IF NOT EXISTS
FOR (n:Entity) REQUIRE n.uuid IS UNIQUE
"""

CREATE_EPISODE_UUID_CONSTRAINT = """
CREATE CONSTRAINT episode_uuid IF NOT EXISTS
FOR (ep:Episode) REQUIRE ep.uuid IS UNIQUE
"""

# Index vectoriels (Neo4j 5.11+)
CREATE_ENTITY_VECTOR_INDEX = """
CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
}}
"""

CREATE_RELATION_VECTOR_INDEX = """
CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
FOR ()-[r:RELATION]-() ON (r.fact_embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
}}
"""

# Index plein texte (pour la recherche par mots-clés BM25)
CREATE_ENTITY_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.summary]
"""

CREATE_FACT_FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
FOR ()-[r:RELATION]-() ON EACH [r.fact, r.name]
"""

# Toutes les requêtes de schéma à exécuter au démarrage
ALL_SCHEMA_QUERIES = [
    CREATE_GRAPH_UUID_CONSTRAINT,
    CREATE_ENTITY_UUID_CONSTRAINT,
    CREATE_EPISODE_UUID_CONSTRAINT,
    CREATE_ENTITY_VECTOR_INDEX,
    CREATE_RELATION_VECTOR_INDEX,
    CREATE_ENTITY_FULLTEXT_INDEX,
    CREATE_FACT_FULLTEXT_INDEX,
]
