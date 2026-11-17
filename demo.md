# Demo script

How to present the demo during *Beyond Vector Memory* (`n.html`). About 5 minutes in
total; steps 1 and 2 carry the core message and can stand alone.

## Before the talk

- Start Neo4j and wait until it is healthy:
  ```bash
  docker compose up -d
  docker ps --format '{{.Names}} {{.Status}}'    # graph-memory-neo4j ... (healthy)
  ```
- Run the demo once to warm everything up: `uv run graph-memory demo`.
- Open Neo4j Browser at http://localhost:7474 (login `neo4j` / `memorydemo`) and
  keep a terminal open with a large font.
- Use the default offline mode (rules extractor, hashing embeddings). It is
  deterministic and needs no network.
- Pre-run the LLM mode and keep the output as a backup. It depends on the zign
  token, the gateway and the model, so do not rely on it live:
  ```bash
  GRAPH_MEMORY_EXTRACTOR=llm uv run graph-memory demo
  ```
- Check the LLM run before relying on it. The model is not deterministic
  (`gpt-6-luna` rejects `temperature=0`), and occasionally a run can omit dates:
  turn 87 then prints `Alex works at Apple` with `valid unknown -> unknown`, and
  Apple is closed on 2026-11-11 (the turn's reference time) instead of
  2026-11-04. Look for `Alex worked at Apple [valid unknown -> 2026-11-04]` in
  the turn-87 output. If it is missing, run the demo again. Relative dates are
  resolved in code and passed to the model as hints, which made 10 of 10 test
  runs correct, but that is not a guarantee. The default `rules` mode always
  gives the output described here.

## During the talk

### 1. Slide "Write path · result" (~1 min)

Run `uv run graph-memory demo` and scroll to turns 32 and 87.

Say: "Apple is closed on 2026-11-04, Neo4j opens the same day, and nothing was deleted."

Switch to Neo4j Browser and run:

```cypher
MATCH p=(:Entity)-[:RELATES_TO]->(:Entity) RETURN p
```

Click the two `WORKS_AT` edges to show `valid_at`, `invalid_at` and `episodes`.
This is the "two clocks" slide in real data.

### 2. Slide "Read path · 4" (~1 min)

Show the turn-88 output, the `<current_facts>` / `<history>` block. This closes
the loop with slide 2, where the agent used the wrong employer.

### 3. Slide "Bi-temporal facts" (~1.5 min)

Show the valid-time query (what was true on 2026-06-01) and the system-time query
(what the agent believed just before turn 32, which still lists Rust).

Say: "I moved changes the world, so valid time. I misspoke changes our knowledge,
so system time."

Caution: `uv run graph-memory asof 2026-06-01` prints "Alex worked at Apple" for a
date when Apple was current. The sentence is rewritten to the past tense when the
fact is closed (see Known limitations in `README.md`). The dates are right. Skip
this command or mention it.

### 4. Optional, for Q&A

```bash
uv run graph-memory add "I moved to Munich last month" --at 2026-11-11
uv run graph-memory search "where do I work"
uv run graph-memory profile
```

`add` writes to the demo graph; run `uv run graph-memory demo` afterwards to reset it.
The rules extractor handles only the demo's sentence patterns; "I moved to Munich
last month" is one of them.

## Expected output

After the demo run:

- Turn 32: the Rust fact is expired and Go replaces it.
- Turn 87: Apple is closed on 2026-11-04 and Neo4j opens from that date.
- Context block: Neo4j under `<current_facts>`, Apple under `<history>`.
- Valid time, 2026-06-01: Apple, Berlin, Go and the allergy; not Neo4j.
- System time, before turn 32: still lists Rust.
- Provenance: omits the corrected Rust fact.

## Questions to be ready for

- **Why does Apple rank above Neo4j in the search results?** The query "where I
  work" is genuinely relevant to both. Similarity cannot tell which is true now,
  so the `CURRENT` / `HISTORY` labels do. This matches the deck: time is a
  label, not a filter.
- **Is the extractor real?** The default one is regexes and only handles the demo
  sentences. Show the LLM-mode output from the backup run
  (`GRAPH_MEMORY_EXTRACTOR=llm`).
- **Are the embeddings semantic?** The default hashing embedder matches on wording
  only. `GRAPH_MEMORY_EMBEDDER=local` uses a sentence-transformers model; the
  ranking was the same on this data.
- **What about other predicates or ambiguous entities?** Only single-valued
  predicates (`WORKS_AT`, `LIVES_IN`, `FAVORITE_LANGUAGE`) are closed
  deterministically. There is no LLM judge yet.

## If something fails

- Neo4j not reachable: check that Docker is running and `docker compose up -d`
  has finished; the CLI retries the connection for about a minute.
- LLM mode errors: fall back to the default mode, or show the backup output.
- Indexes out of sync after changing the embedder or `EMBEDDING_DIM`:
  `uv run graph-memory reset`, then run the demo again.
