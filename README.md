# Graph memory demo

A runnable version of the talk *Beyond Vector Memory: temporal graph memory for
conversational agents with Neo4j* (`n.html`). An agent's memory is a bi-temporal
knowledge graph: entities are nodes, facts are edges with two clocks, and old
facts are invalidated rather than deleted.

## Run it

```bash
docker compose up -d          # Neo4j 5.26 on :7474 (Browser) and :7687 (Bolt)
uv run graph-memory demo      # resets the DB, ingests a scripted conversation, queries it
```

Needs Python 3.14 (uv fetches it), Docker, and no API key in the default mode.

To use a real LLM extractor and semantic embeddings:

```bash
uv sync --extra local                                  # adds sentence-transformers (pulls torch)
GRAPH_MEMORY_EXTRACTOR=llm GRAPH_MEMORY_EMBEDDER=local uv run graph-memory demo
```

Then open http://localhost:7474 (login `neo4j` / `memorydemo`) and run
`MATCH p=(:Entity)-[:RELATES_TO]->(:Entity) RETURN p` to see the graph.

Other commands (they act on the demo user "Alex"):

```bash
uv run graph-memory add "I moved to Munich last month" --at 2026-11-11
uv run graph-memory profile               # current facts
uv run graph-memory asof 2026-06-01       # what was true on a date
uv run graph-memory search "where do I work"
uv run graph-memory reset
```

## Configuration

Settings come from environment variables or a `.env` file in the working
directory (real environment variables win). See `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | `bolt://localhost:7687`, `neo4j`, `memorydemo`, `neo4j` | Database connection |
| `GRAPH_MEMORY_EXTRACTOR` | `rules` | `rules` (regexes, offline) or `llm` (structured output via the gateway) |
| `GRAPH_MEMORY_EMBEDDER` | `hashing` | `hashing` (lexical, offline) or `local` (sentence-transformers) |
| `EMBEDDING_DIM` | `384` | Must match the embedder and the vector indexes; `reset` after changing it |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for the `local` embedder (384 dims) |
| `OPENAI_BASE_URL` | `` | OpenAI-compatible gateway for the `llm` extractor |
| `OPENAI_MODEL` | `gpt-6-luna` | Extraction model; |
| `OPENAI_API_KEY` | unset | Optional. |

The gateway has no embedding model, which is why embeddings are local.
`temperature` is not set because `gpt-6-luna` rejects `0`.

## What the demo shows

| Deck slide | Where |
|---|---|
| Graph schema, constraints and indexes | `schema.py` |
| Write path 1: store the episode | `GraphMemory.add_episode` |
| Write path 2: extract entities and facts | `extraction.py` |
| Write path 3: entity resolution (exact, vector, fulltext) | `GraphMemory.resolve_entity` |
| Write path 4: temporal upsert and invalidation | `GraphMemory.upsert_fact` |
| Corrections use `expired_at` (system time) | `GraphMemory.ingest_episode` |
| Read path 1: current profile, point-in-time | `current_facts`, `facts_as_of` |
| Audit: what did the agent believe at time T | `beliefs_at` |
| Read path 2: hybrid retrieval in one Cypher query | `_semantic_and_graph`, `_bm25` |
| Read path 3: RRF fusion | `rrf` |
| Read path 4: context assembly | `render_context` |
| Right to be forgotten | `forget_episode` |

| File | Role |
|---|---|
| `memory.py` | `GraphMemory`: write path, read paths, `rrf`, `render_context` |
| `extraction.py` | `Extractor` protocol, `RuleBasedExtractor`, `LLMExtractor`, predicate allowlist |
| `embeddings.py` | `HashingEmbedder`, `LocalEmbedder` |
| `llm_client.py` | OpenAI-compatible client with zign token |
| `schema.py` | Constraints and indexes, `reset` |
| `config.py` | Settings and `.env` loading |
| `cli.py` | Commands and the scripted demo |

Expected demo result: after turn 87, Neo4j is under `<current_facts>` and Apple
under `<history>`; the June 2026 point-in-time query shows Apple, not Neo4j; the
audit query before turn 32 still shows Rust; provenance omits the corrected Rust fact.

## Differences from the deck

- **Extractor and embedder are chosen separately.**
  - `GRAPH_MEMORY_EXTRACTOR=rules` (default) uses `RuleBasedExtractor`, which only
    understands sentences like the demo's. `llm` uses `LLMExtractor` (structured
    output) through the gateway at `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the model in `OPENAI_MODEL`.
    Only predicates in `PREDICATE_TYPES` become edges (the deck's poisoning defence).
  - `GRAPH_MEMORY_EMBEDDER=hashing` (default) matches on wording, not meaning.
    `local` uses a sentence-transformers model (`uv sync --extra local`). The deck
    uses 1536 dims; the gateway has no embedding model, so the local default is 384.
  - After changing the embedder or `EMBEDDING_DIM`, run `reset` to rebuild the indexes.
- **Ambiguous entity matches create a new entity** instead of calling an LLM judge.
- **`observed_at` on each fact.** When a fact has no start date, the turn's
  reference time stands in for it. Without this, late-arriving history (an
  older dated fact) would close a current undated one.
- **Fact identity** is (subject, predicate, object, valid_at), so repeating a
  fact or retrying a turn adds provenance instead of a duplicate edge.
- **Fulltext uses the English analyzer**, so "work" matches "works" and "worked".
- **Entity identity includes type**, so "apple" (allergen) and "Apple" (employer)
  are different entities. The entity uuid is derived from group, type and name.
- **A fact that is re-asserted after it was closed is reopened**
  (Berlin, then Paris, then Berlin again leaves Berlin current).
- **Graph distance joins the RRF fusion only when facts differ in distance.**
  In the demo every fact is about the user (distance 0), so it would just repeat
  the vector ranking; it starts to matter once facts about other entities exist.
- **Vector searches over-fetch** (50 entities, 150 facts) before filtering by
  `group_id`, so other tenants don't crowd out the user's results.

## Known limitations

- **Ranking does not separate current from historical.** For "where do I work",
  "Alex worked at Apple" can rank just above "Alex works at Neo4j", with either
  embedder. That follows the deck: time is a label, not a filter, and
  `render_context` puts each fact under `<current_facts>` or `<history>`.
- **Tense is stored in the sentence.** When a repeat says a fact has ended, the
  stored sentence is rewritten to the past tense. Point-in-time and audit queries
  for earlier dates then show "worked at" although the fact was current then.
- Contradictions are only detected for single-valued predicates
  (`WORKS_AT`, `LIVES_IN`, `FAVORITE_LANGUAGE`). There is no LLM judge for other
  predicates or ambiguous entity matches, no `search_memory` tool, and no MMR or
  cross-encoder reranking.
- Only the predicates in `PREDICATE_TYPES` become edges; extend that dict to add more.
- The gateway token is fetched when a command starts; a long-lived process would
  need to rebuild the client when it expires.
- `uv.lock` may be stale after the `openai` and `stups-zign` dependencies were
  added; run `uv lock`.
- `forget_episode` deletes facts with no remaining provenance, but does not
  reopen facts those deleted facts had closed.
- `beliefs_at` uses `created_at`/`expired_at` only; it doesn't track when an
  edge's `invalid_at` was set.
- Each write runs as a separate query, not in one transaction, and in the
  calling thread. Production would run the write path async, one user at a time.
