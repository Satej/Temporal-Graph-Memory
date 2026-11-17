"""Constraints and indexes (deck: "Constraints and indexes")."""

from neo4j import Driver


def schema_statements(dim: int) -> list[str]:
    vec = f"{{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
    return [
        "CREATE CONSTRAINT episode_uuid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uuid IS UNIQUE",
        "CREATE CONSTRAINT entity_uuid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE",
        "CREATE INDEX entity_group IF NOT EXISTS FOR (n:Entity) ON (n.group_id, n.norm_name)",
        "CREATE INDEX fact_key IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.key)",
        f"CREATE VECTOR INDEX entity_name_embedding IF NOT EXISTS FOR (n:Entity) ON n.name_embedding OPTIONS {vec}",
        # Relationship vector index: search over facts directly
        f"CREATE VECTOR INDEX fact_embedding IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON r.fact_embedding OPTIONS {vec}",
        "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.summary]",
        # English analyzer stems "work" / "works" / "worked" to one term
        "CREATE FULLTEXT INDEX fact_text IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON EACH [r.fact]"
        " OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
    ]


def ensure_schema(driver: Driver, database: str, dim: int) -> None:
    for stmt in schema_statements(dim):
        driver.execute_query(stmt, database_=database)
    driver.execute_query("CALL db.awaitIndexes(60)", database_=database)


def reset(driver: Driver, database: str) -> None:
    """Drop all data, indexes and constraints (demo only)."""
    driver.execute_query("MATCH (n) DETACH DELETE n", database_=database)
    for name in ("entity_name_embedding", "fact_embedding", "entity_text", "fact_text",
                 "entity_group", "fact_key"):
        driver.execute_query(f"DROP INDEX {name} IF EXISTS", database_=database)
    for name in ("episode_uuid", "entity_uuid"):
        driver.execute_query(f"DROP CONSTRAINT {name} IF EXISTS", database_=database)
