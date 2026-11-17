import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()

@dataclass(frozen=True)
class Settings:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "memorydemo")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    # extractor: "rules" (offline regexes) or "llm" (structured output via the gateway)
    extractor: str = os.getenv("GRAPH_MEMORY_EXTRACTOR", "rules")
    # embedder: "hashing" (offline, lexical) or "local" (sentence-transformers, semantic)
    embedder: str = os.getenv("GRAPH_MEMORY_EMBEDDER", "hashing")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
