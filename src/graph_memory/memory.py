"""Temporal graph memory on Neo4j: write path and read path from the deck."""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

from neo4j import Driver

from graph_memory.config import settings
from graph_memory.embeddings import Embedder, HashingEmbedder
from graph_memory.extraction import (
    PREDICATE_TYPES,
    SINGLE_VALUED,
    ExtractedFact,
    Extractor,
    RuleBasedExtractor,
)


# Vector indexes filter by group_id after top-k, so over-fetch to leave room for other tenants.
ENTITY_K = 50
FACT_K = 150


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def lucene_escape(text: str) -> str:
    """Turn free text into a safe fulltext (Lucene) query: plain OR of words.

    Lowercased because Lucene only treats uppercase AND / OR / NOT as operators.
    """
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def is_zero(vec: list[float]) -> bool:
    """A zero vector (no embeddable tokens) is rejected by cosine vector indexes."""
    return not any(vec)


def fmt(value) -> str:
    return "unknown" if value is None else str(value)[:10]


@dataclass
class IngestReport:
    episode_uuid: str
    resolutions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)


class GraphMemory:
    def __init__(self, driver: Driver, group_id: str, database: str = "neo4j",
                 embedder: Embedder | None = None, extractor: Extractor | None = None):
        self.driver = driver
        self.group_id = group_id
        self.database = database
        self.embedder = embedder or HashingEmbedder(settings.embedding_dim)
        self.extractor = extractor or RuleBasedExtractor()

    def _q(self, query: str, **params):
        records, _, _ = self.driver.execute_query(
            query, group_id=self.group_id, database_=self.database, **params)
        return records

    # ------------------------------------------------------------------ write path

    def add_episode(self, content: str, reference_time: datetime, source: str = "chat") -> str:
        """Step 1: store the raw turn, immutable, linked to the previous one."""
        ep_uuid = str(uuid4())
        self._q("""
            OPTIONAL MATCH (prev:Episode {group_id: $group_id})
            WHERE NOT (prev)-[:NEXT]->()
            WITH prev ORDER BY prev.reference_time DESC LIMIT 1
            CREATE (e:Episode {uuid: $uuid, group_id: $group_id, content: $content,
                               source: $source, reference_time: $ref_time, created_at: $now})
            FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
                     CREATE (prev)-[:NEXT]->(e))
        """, uuid=ep_uuid, content=content, source=source, ref_time=reference_time,
             now=datetime.now(UTC))
        return ep_uuid

    def resolve_entity(self, name: str, type_: str, summary: str = "") -> tuple[str, str]:
        """Step 3: map a name to an existing entity, or create one.

        Returns (uuid, decision) where decision explains what happened.
        """
        n = norm(name)
        # Same name, different type ("apple" the allergen vs "Apple" the employer)
        # are different entities, so type is part of both the match and the uuid.
        exact = self._q("""
            MATCH (n:Entity {group_id: $group_id, norm_name: $norm, type: $type})
            RETURN n.uuid AS uuid LIMIT 1
        """, norm=n, type=type_)
        if exact:
            return exact[0]["uuid"], f"'{name}' -> exact match"

        emb = self.embedder.embed(name)
        strong: dict[str, dict] = {}
        if not is_zero(emb):
            candidates = self._q("""
                CALL db.index.vector.queryNodes('entity_name_embedding', $k, $name_emb)
                YIELD node, score
                WHERE node.group_id = $group_id AND node.type = $type AND score >= 0.85
                RETURN node.uuid AS uuid, node.name AS name, score
                ORDER BY score DESC LIMIT 10
            """, k=ENTITY_K, name_emb=emb, type=type_)
            strong = {c["uuid"]: c for c in candidates}
        if len(strong) == 1:
            c = next(iter(strong.values()))
            return c["uuid"], f"'{name}' -> merged into '{c['name']}' (vector {c['score']:.2f})"
        # Ambiguous: the deck sends these to a small LLM judge. The demo creates a
        # new entity instead, which is the safe default.

        uuid = str(uuid5(NAMESPACE_URL, f"{self.group_id}/{type_}/{n}"))
        self._q("""
            MERGE (n:Entity {uuid: $uuid})
            ON CREATE SET n.group_id = $group_id, n.name = $name, n.norm_name = $norm,
                          n.type = $type, n.summary = $summary, n.name_embedding = $emb
        """, uuid=uuid, name=name, norm=n, type=type_, summary=summary,
             emb=None if is_zero(emb) else emb)
        why = "ambiguous, new entity" if len(strong) > 1 else "new entity"
        return uuid, f"'{name}' -> {why}"

    def upsert_fact(self, subject_uuid: str, object_uuid: str, f: ExtractedFact,
                    episode_uuid: str, observed_at: datetime) -> dict:
        """Step 4: temporal upsert and invalidation.

        Facts are keyed by (subject, predicate, object, valid_at), so repeating a
        fact or retrying an episode attaches provenance instead of duplicating.

        When a start date is unknown, the turn's reference time (`observed_at`)
        stands in for it: "I live in Paris" said in November is known true by
        November. Without this, an undated fact sorts before every dated one
        and late-arriving history would wrongly close it.
        """
        key = hashlib.sha1(
            f"{self.group_id}|{subject_uuid}|{f.predicate}|{object_uuid}|{f.valid_at}".encode()
        ).hexdigest()
        rec = self._q("""
            MATCH (s:Entity {uuid: $subject_uuid}), (o:Entity {uuid: $object_uuid})
            // 1. Close current facts with the same single-valued predicate
            CALL (s, o) {
              MATCH (s)-[old:RELATES_TO {predicate: $predicate}]->(prev:Entity)
              WHERE $single AND $invalid_at IS NULL AND prev <> o
                AND old.invalid_at IS NULL AND old.expired_at IS NULL
                AND coalesce(old.valid_at, old.observed_at) <= $effective_start
              SET old.invalid_at = $effective_start
            }
            // 2. Late history: a newer value already exists, so create this one closed
            CALL (s, o) {
              OPTIONAL MATCH (s)-[later:RELATES_TO {predicate: $predicate}]->(x:Entity)
              WHERE $single AND x <> o AND later.expired_at IS NULL
                AND coalesce(later.valid_at, later.observed_at) > $effective_start
              RETURN min(coalesce(later.valid_at, later.observed_at)) AS later_start
            }
            // 3. Create the fact, or attach provenance to the existing one
            MERGE (s)-[r:RELATES_TO {key: $key}]->(o)
              ON CREATE SET r.uuid = randomUUID(), r.group_id = $group_id,
                r.predicate = $predicate, r.fact = $fact, r.fact_embedding = $fact_emb,
                r.valid_at = $valid_at, r.observed_at = $observed_at,
                r.invalid_at = coalesce($invalid_at, later_start),
                r.created_at = $now, r.episodes = [$episode_uuid]
              ON MATCH SET
                r.episodes = CASE WHEN $episode_uuid IN r.episodes THEN r.episodes
                                  ELSE r.episodes + $episode_uuid END,
                // A repeat that says the fact has ended ("used to work at") rewrites the
                // sentence to the past tense and re-embeds it. Must precede invalid_at below.
                r.fact = CASE WHEN $invalid_at IS NOT NULL AND r.invalid_at IS NULL
                              THEN $fact ELSE r.fact END,
                r.fact_embedding = CASE WHEN $invalid_at IS NOT NULL AND r.invalid_at IS NULL
                                        THEN $fact_emb ELSE r.fact_embedding END,
                // Re-asserted after being closed (Berlin -> Paris -> Berlin): reopen it,
                // unless this assertion carries its own end or a newer value exists.
                r.invalid_at = CASE
                  WHEN r.invalid_at IS NOT NULL AND $invalid_at IS NULL AND later_start IS NULL
                       AND $effective_start >= r.invalid_at THEN null
                  ELSE coalesce(r.invalid_at, $invalid_at) END
            RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at
        """, subject_uuid=subject_uuid, object_uuid=object_uuid, predicate=f.predicate,
             single=f.predicate in SINGLE_VALUED, fact=f.fact,
             fact_emb=self.embedder.embed(f.fact), valid_at=f.valid_at,
             invalid_at=f.invalid_at, observed_at=observed_at,
             effective_start=f.valid_at or observed_at, key=key, episode_uuid=episode_uuid,
             now=datetime.now(UTC))
        return dict(rec[0])

    def _find_corrected(self, subject_uuid: str, old_name: str) -> dict | None:
        rec = self._q("""
            MATCH (s:Entity {uuid: $s})-[r:RELATES_TO]->(o:Entity {group_id: $group_id})
            WHERE o.norm_name = $old AND r.expired_at IS NULL
            RETURN r.uuid AS uuid, r.predicate AS predicate, r.fact AS fact,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at, o.name AS name
            ORDER BY r.created_at DESC LIMIT 1
        """, s=subject_uuid, old=norm(old_name))
        return dict(rec[0]) if rec else None

    def ingest_episode(self, content: str, reference_time: datetime,
                       user_name: str) -> IngestReport:
        """Full write path for one turn. In production this runs async, off the hot path."""
        previous = [r["content"] for r in self._q("""
            MATCH (e:Episode {group_id: $group_id})
            RETURN e.content AS content ORDER BY e.reference_time DESC LIMIT 3
        """)]
        ep_uuid = self.add_episode(content, reference_time)
        report = IngestReport(ep_uuid)
        extraction = self.extractor.extract(content, reference_time, user_name, previous)

        user_uuid, _ = self.resolve_entity(user_name, "Person", "The user")
        corrections: dict[int, dict] = {}
        for i, f in enumerate(extraction.facts):
            if f.corrects and (old := self._find_corrected(user_uuid, f.corrects)):
                corrections[i] = old
                f.predicate = old["predicate"]
                f.fact = re.sub(re.escape(old["name"]), f.object, old["fact"], flags=re.I)
                f.valid_at, f.invalid_at = old["valid_at"], old["invalid_at"]

        uuids: dict[str, str] = {}
        types = {f.object: PREDICATE_TYPES.get(f.predicate, "Thing") for f in extraction.facts}
        for ent in extraction.entities:
            if ent.name in uuids:
                continue
            uuid, decision = self.resolve_entity(ent.name, types.get(ent.name, ent.type),
                                                 ent.summary)
            uuids[ent.name] = uuid
            report.resolutions.append(decision)
            self._q("""
                MATCH (e:Episode {uuid: $ep}), (n:Entity {uuid: $n}) MERGE (e)-[:MENTIONS]->(n)
            """, ep=ep_uuid, n=uuid)

        for i, f in enumerate(extraction.facts):
            if not f.predicate:
                report.facts.append(f"(skipped correction: nothing found for '{f.corrects}')")
                continue
            if i in corrections:
                # "I misspoke" changes our knowledge, not the world: system time.
                # Expire first so the new value doesn't also close it in valid time.
                self._q("""
                    MATCH ()-[old:RELATES_TO {uuid: $uuid}]->() SET old.expired_at = $now
                """, uuid=corrections[i]["uuid"], now=datetime.now(UTC))
                report.facts.append(f"expired: {corrections[i]['fact']}")
            r = self.upsert_fact(uuids[f.subject], uuids[f.object], f, ep_uuid, reference_time)
            report.facts.append(
                f"{r['fact']}  [valid {fmt(r['valid_at'])} -> {fmt(r['invalid_at'])}]")
        return report

    def forget_episode(self, episode_uuid: str) -> int:
        """Right to be forgotten: delete an episode, then facts with no provenance left."""
        self._q("""
            MATCH (p:Episode)-[:NEXT]->(e:Episode {uuid: $ep, group_id: $group_id})-[:NEXT]->(n)
            CREATE (p)-[:NEXT]->(n)
        """, ep=episode_uuid)
        self._q("MATCH (e:Episode {uuid: $ep, group_id: $group_id}) DETACH DELETE e",
                ep=episode_uuid)
        rec = self._q("""
            MATCH ()-[r:RELATES_TO {group_id: $group_id}]->() WHERE $ep IN r.episodes
            SET r.episodes = [x IN r.episodes WHERE x <> $ep]
            WITH r WHERE size(r.episodes) = 0
            DELETE r
            RETURN count(*) AS deleted
        """, ep=episode_uuid)
        self._q("""
            MATCH (n:Entity {group_id: $group_id})
            WHERE NOT (n)-[:RELATES_TO]-() AND NOT ()-[:MENTIONS]->(n)
            DELETE n
        """)
        return rec[0]["deleted"]

    # ------------------------------------------------------------------- read path

    def user_uuid(self, user_name: str) -> str | None:
        rec = self._q("MATCH (n:Entity {group_id: $group_id, norm_name: $n, type: 'Person'}) "
                      "RETURN n.uuid AS u", n=norm(user_name))
        return rec[0]["u"] if rec else None

    def current_facts(self, user_uuid: str, limit: int = 25) -> list[dict]:
        """Read path 1: the 1-hop user profile. Cheap, deterministic, no embeddings."""
        return [dict(r) for r in self._q("""
            MATCH (u:Entity {uuid: $user_uuid})-[r:RELATES_TO]-(n:Entity)
            WHERE r.invalid_at IS NULL AND r.expired_at IS NULL
            RETURN r.fact AS fact, r.predicate AS predicate, n.name AS about,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at, true AS is_current
            ORDER BY coalesce(r.valid_at, r.observed_at) DESC LIMIT $limit
        """, user_uuid=user_uuid, limit=limit)]

    def facts_as_of(self, user_uuid: str, as_of: datetime) -> list[dict]:
        """Valid time: what was true in the world at `as_of`."""
        return [dict(r) for r in self._q("""
            MATCH (u:Entity {uuid: $user_uuid})-[r:RELATES_TO]-(n:Entity)
            WHERE r.expired_at IS NULL
              AND coalesce(r.valid_at, r.observed_at) <= $as_of
              AND (r.invalid_at IS NULL OR r.invalid_at > $as_of)
            RETURN r.fact AS fact, r.predicate AS predicate, n.name AS about,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at
            ORDER BY r.predicate
        """, user_uuid=user_uuid, as_of=as_of)]

    def beliefs_at(self, user_uuid: str, system_time: datetime) -> list[dict]:
        """System time: which facts the agent held at `system_time` (audit/debugging)."""
        return [dict(r) for r in self._q("""
            MATCH (u:Entity {uuid: $user_uuid})-[r:RELATES_TO]-(n:Entity)
            WHERE r.created_at <= $t AND (r.expired_at IS NULL OR r.expired_at > $t)
            RETURN r.fact AS fact, r.created_at AS created_at, r.expired_at AS expired_at
            ORDER BY r.created_at
        """, user_uuid=user_uuid, t=system_time)]

    def provenance(self, user_uuid: str) -> list[dict]:
        """Which turns support each fact the agent still believes (corrected facts are skipped)."""
        return [dict(r) for r in self._q("""
            MATCH (u:Entity {uuid: $user_uuid})-[r:RELATES_TO]-(:Entity)
            WHERE r.expired_at IS NULL
            UNWIND r.episodes AS ep_uuid
            MATCH (e:Episode {uuid: ep_uuid})
            RETURN r.fact AS fact, collect(e.content) AS said_in
        """, user_uuid=user_uuid)]

    def _semantic_and_graph(self, query_emb: list[float], user_uuid: str) -> list[dict]:
        """Read path 2: vector seeds + graph expansion + distance to the user, one query."""
        return [dict(r) for r in self._q("""
            MATCH (u:Entity {uuid: $user_uuid})
            // 1. Semantic seeds: facts close to the query
            CALL db.index.vector.queryRelationships('fact_embedding', $k, $query_emb)
            YIELD relationship AS r, score
            WHERE r.group_id = $group_id AND r.expired_at IS NULL
            WITH u, r, score ORDER BY score DESC LIMIT 30
            WITH u, r, score, startNode(r) AS s, endNode(r) AS e
            // 2. Graph expansion: current neighbours of the seed's subject
            OPTIONAL MATCH (s)-[nb:RELATES_TO]-(:Entity)
            WHERE nb.invalid_at IS NULL AND nb.expired_at IS NULL AND nb <> r
            WITH u, r, score, s, e, collect(DISTINCT nb)[0..5] AS nbs
            // 3. Proximity to the user: nearer of the fact's two endpoints
            CALL (u, s, e) {
              UNWIND [s, e] AS x
              WITH u, x WHERE u <> x
              MATCH p = shortestPath((u)-[:RELATES_TO*..3]-(x))
              RETURN min(length(p)) AS hops
            }
            RETURN r.uuid AS uuid, r.fact AS fact,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                   r.invalid_at IS NULL AS is_current,
                   score AS semantic_score,
                   CASE WHEN u = s OR u = e THEN 0 ELSE coalesce(hops, 4) END AS hops_from_user,
                   [x IN nbs | x.fact] AS context
            ORDER BY semantic_score DESC
        """, user_uuid=user_uuid, query_emb=query_emb, k=FACT_K)]

    def _bm25(self, query: str) -> list[dict]:
        q = lucene_escape(query)
        if not q:
            return []
        return [dict(r) for r in self._q("""
            CALL db.index.fulltext.queryRelationships('fact_text', $q)
            YIELD relationship AS r, score
            WHERE r.group_id = $group_id AND r.expired_at IS NULL
            RETURN r.uuid AS uuid, r.fact AS fact, r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at, r.invalid_at IS NULL AS is_current,
                   score AS bm25_score
            ORDER BY score DESC LIMIT 30
        """, q=q)]

    def hybrid_search(self, query: str, user_uuid: str, limit: int = 8) -> list[dict]:
        """Vector + BM25 + graph distance, fused with RRF. Time is a label, not a filter."""
        query_emb = self.embedder.embed(query)
        vec = [] if is_zero(query_emb) else self._semantic_and_graph(query_emb, user_uuid)
        bm25 = self._bm25(query)
        ranked = [[r["uuid"] for r in vec], [r["uuid"] for r in bm25]]
        # Distance only adds a signal when facts differ in distance; otherwise it
        # would just repeat the vector ranking and count it twice.
        if len({r["hops_from_user"] for r in vec}) > 1:
            by_distance = sorted(vec, key=lambda r: (r["hops_from_user"], -r["semantic_score"]))
            ranked.append([r["uuid"] for r in by_distance])
        fused = rrf(*ranked)
        rows = {r["uuid"]: r for r in bm25} | {r["uuid"]: r for r in vec}
        return [rows[u] | {"rrf": s} for u, s in fused[:limit]]


def rrf(*ranked: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: free, robust default reranker."""
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked:
        for rank, uid in enumerate(lst):
            scores[uid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def render_context(facts: list[dict]) -> str:
    """Read path 4: label facts so history informs, but doesn't override, the present."""
    cur = [f for f in facts if f["invalid_at"] is None]
    his = [f for f in facts if f["invalid_at"] is not None]
    out = ["<current_facts>"]
    out += [f"- {f['fact']} (since {fmt(f['valid_at'])})" for f in cur]
    out += ["</current_facts>", "<history>"]
    out += [f"- {f['fact']} ({fmt(f['valid_at'])} – {fmt(f['invalid_at'])})" for f in his]
    out += ["</history>"]
    return "\n".join(out)
