"""Command line entry point: `uv run graph-memory demo`."""

import argparse
import sys
import time
from datetime import UTC, datetime

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from graph_memory.config import settings
from graph_memory.embeddings import HashingEmbedder, LocalEmbedder
from graph_memory.extraction import LLMExtractor
from graph_memory.memory import GraphMemory, fmt, render_context
from graph_memory.schema import ensure_schema, reset

GROUP = "user-alex"
USER = "Alex"

# (turn, reference time, what the user said)
SCRIPT = [
    (3, "2026-03-02", "I'm allergic to peanuts, please keep that in mind."),
    (12, "2026-03-10", "I work at Apple on the Maps team."),
    (20, "2026-04-02", "I live in Berlin."),
    (31, "2026-05-14", "My favourite language is Rust."),
    (32, "2026-05-14", "Sorry, I meant Go, not Rust."),
    (87, "2026-11-11", "I used to work at Apple, but last week I started at Neo4j."),
]
QUERY = "Draft an intro email to a new colleague about where I work."

BOLD, DIM, TEAL, AMBER, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"


def h(title: str) -> None:
    print(f"\n{BOLD}{TEAL}== {title}{RESET}")


def ts(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=9, tzinfo=UTC)


def connect():
    driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password),
                                  notifications_min_severity="OFF")
    for _ in range(30):
        try:
            driver.verify_connectivity()
            return driver
        except ServiceUnavailable:
            time.sleep(2)
    sys.exit(f"Could not reach Neo4j at {settings.uri}. Is `docker compose up -d` running?")


def memory(driver) -> GraphMemory:
    if settings.embedder == "local":
        embedder = LocalEmbedder(settings.embedding_dim, settings.embedding_model)
    else:
        embedder = HashingEmbedder(settings.embedding_dim)
    extractor = LLMExtractor(settings.llm_model) if settings.extractor == "llm" else None
    return GraphMemory(driver, GROUP, settings.database, embedder, extractor)


def run_demo(driver) -> None:
    reset(driver, settings.database)
    ensure_schema(driver, settings.database, settings.embedding_dim)
    mem = memory(driver)

    h("Write path: ingest the conversation")
    before_correction = None
    for turn, day, text in SCRIPT:
        if turn == 32:
            before_correction = datetime.now(UTC)
            time.sleep(0.05)
        rep = mem.ingest_episode(text, ts(day), USER)
        print(f"\n{BOLD}Turn {turn} · {day}{RESET}  \"{text}\"")
        for r in rep.resolutions:
            print(f"  {DIM}resolve  {r}{RESET}")
        for f in rep.facts:
            print(f"  fact     {f}")

    user = mem.user_uuid(USER)

    h(f"Turn 88 · read path · \"{QUERY}\"")
    print(f"{DIM}Hybrid search (vector + BM25 + graph distance, RRF fused):{RESET}")
    results = mem.hybrid_search(QUERY, user)
    for r in results:
        label = f"{TEAL}CURRENT{RESET}" if r["is_current"] else f"{AMBER}HISTORY{RESET}"
        print(f"  {label}  rrf={r['rrf']:.4f}  {r['fact']}")

    h("Context handed to the LLM")
    print(render_context(results))
    print(f"{DIM}A vector-only memory sees 'works at Apple' and 'works at Neo4j' as equally"
          f" relevant. Here Apple is labelled history, so the reply says Neo4j.{RESET}")

    h("Profile: current facts about Alex (no embeddings)")
    for f in mem.current_facts(user):
        print(f"  - {f['fact']} (since {fmt(f['valid_at'])})")

    h("Valid time: what was true on 2026-06-01?")
    for f in mem.facts_as_of(user, ts("2026-06-01")):
        print(f"  - {f['fact']}")

    h("System time: what did the agent believe just before turn 32?")
    for f in mem.beliefs_at(user, before_correction):
        print(f"  - {f['fact']}")
    print(f"{DIM}'Rust' was a misstatement, so it is expired (system time), not invalidated"
          f" (valid time).{RESET}")

    h("Provenance: which turns produced each fact")
    for p in mem.provenance(user):
        print(f"  - {p['fact']}")
        for s in p["said_in"]:
            print(f"      {DIM}<- \"{s}\"{RESET}")

    print(f"\nOpen {BOLD}http://localhost:7474{RESET} (neo4j / {settings.password}) and run:")
    print("  MATCH p=(:Entity)-[:RELATES_TO]->(:Entity) RETURN p")


def main() -> None:
    p = argparse.ArgumentParser(prog="graph-memory", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="reset the database and run the scripted demo")
    sub.add_parser("reset", help="delete all data and indexes")
    sub.add_parser("profile", help="show current facts about the user")
    add = sub.add_parser("add", help="ingest one user turn")
    add.add_argument("text")
    add.add_argument("--at", help="reference date, YYYY-MM-DD (default: now)")
    s = sub.add_parser("search", help="hybrid search over memory")
    s.add_argument("query")
    a = sub.add_parser("asof", help="facts that were true on a date")
    a.add_argument("date", help="YYYY-MM-DD")
    args = p.parse_args()

    driver = connect()
    try:
        mem = memory(driver)
        match args.cmd:
            case "demo":
                run_demo(driver)
            case "reset":
                reset(driver, settings.database)
                print("Database cleared.")
            case "add":
                ensure_schema(driver, settings.database, settings.embedding_dim)
                ref = ts(args.at) if args.at else datetime.now(UTC)
                rep = mem.ingest_episode(args.text, ref, USER)
                print("\n".join(rep.resolutions + rep.facts) or "No facts extracted.")
            case "profile":
                for f in mem.current_facts(mem.user_uuid(USER) or ""):
                    print(f"- {f['fact']} (since {fmt(f['valid_at'])})")
            case "search":
                print(render_context(mem.hybrid_search(args.query, mem.user_uuid(USER) or "")))
            case "asof":
                for f in mem.facts_as_of(mem.user_uuid(USER) or "", ts(args.date)):
                    print(f"- {f['fact']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
