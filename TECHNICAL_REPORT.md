# Context Engineering with a Temporal Knowledge Graph for LLM Conversational Memory

**A System Description and Evaluation**

*Technical Report — July 2026*

---

## Abstract

Large language models are stateless across turns: without an external memory, a
conversational agent forgets facts the user stated moments earlier, and cannot
detect when a new statement *contradicts* an older one. We describe a compact
memory system that stores conversational facts as a **temporal knowledge graph
(TKG)** and engineers them into a grounded system prompt on every turn. Facts
are extracted from user messages (by regular-expression patterns or, optionally,
by an LLM), written to a directed multigraph with per-edge event timestamps, and
retrieved by a concept-expansion query layer that maps category words
("database", "caching") to concrete entities. A context-engineering stage then
filters retrieved facts for relevance, resolves temporal conflicts using a
relation-cardinality rule (single-valued relations are *superseded* by newer
values; multi-valued relations *coexist*), compresses to a fixed budget, and
assembles a structured prompt. On a six-message reference dialogue the system
correctly answers category questions without literal keyword overlap, keeps the
latest project focus while dropping superseded ones, and preserves multiple
simultaneous facts of the same type. We report the system's behavior across 16
characterization checks — including four probes that document genuine failure
modes (out-of-vocabulary entities, negation, mixed-intent sentences) — and
position the work honestly relative to existing agent-memory systems. **This is
an engineering system description, not a novel-contribution research paper**; we
close with the specific angles that would need empirical work to become one.

---

## 1. Introduction

An LLM assistant that "remembers" must solve three problems: **(i) extraction** —
turning free-text messages into structured facts; **(ii) storage with time** —
recording *when* each fact became true, so later contradictions can be
adjudicated; and **(iii) retrieval and assembly** — surfacing the right facts for
the current turn and packing them into the model's context window. Retrieval-
augmented generation (RAG) addresses (iii) for static documents, but
conversational memory adds a temporal dimension: the same subject–relation pair
can take different values over time ("I'm building a REST API" → "let's switch to
GraphQL"), and the memory must return the *current* value, not all of them.

This report documents a small, self-contained implementation of such a system.
It is built on a NetworkX `MultiDiGraph` and runs as a command-line demonstration
with a fully transparent, layer-by-layer trace of extraction, retrieval, and
prompt construction. Our goal here is not to claim a new algorithm but to
describe the design precisely, report what it actually does (correct behaviors
*and* failure modes) from direct measurement, and situate it in the landscape of
existing memory systems.

---

## 2. System Architecture

The pipeline is organized as six layers, executed on every user turn:

```
User message
   │
   ▼
[1] Entity / relation extraction   ──►  facts {entities, relationships}
   │
   ▼
[2] TKG write                      ──►  timestamped nodes + edges
   │
   ▼
[3] Retrieval (concept expansion)  ──►  candidate facts, recency-sorted
   │
   ▼
[4] Context engineering            ──►  relevance filter → conflict
   │                                     resolution → compression
   ▼
[5] System-prompt assembly         ──►  structured, sectioned prompt
   │
   ▼
[6] LLM response (optional)        ──►  grounded answer
```

**Separation of concerns.** Layers 1–3 (`tkg_core.py`, class `TKGMemory`) own the
graph. Layers 4–5 (`context_engineer.py`, class `ContextEngineer`) own prompt
construction and never mutate the graph. Layer 6 (`tkg_cli.py`) is a thin driver
that either calls an LLM or produces a deterministic simulated answer, so the
memory machinery can be exercised and tested without any API access.

---

## 3. Temporal Data Model

### 3.1 Nodes and edges

The graph is a directed multigraph. The user is a single fixed node
(`user_demo`). Every extracted entity becomes a typed node (`Technology`,
`Constraint`, …) carrying `first_mentioned` and `last_mentioned` timestamps and
the id of the message that introduced it. Every relationship becomes an edge
carrying:

| Field | Meaning |
|-------|---------|
| `type` | relation label (`WORKING_ON`, `USES`, `HAS_PREFERENCE`, `HAS_CONSTRAINT`) |
| `event_time` | when the fact became true (message timestamp) |
| `ingest_time` | when it was written (equal to `event_time` here) |
| `message_id` | monotonic message counter — a **tiebreaker** for equal timestamps |
| `confidence` | fixed at 0.9 |

The `message_id` tiebreaker is not incidental. On Windows, `datetime.now()` has
coarse (~15 ms) resolution, so two messages ingested in a tight loop can receive
*identical* ISO timestamps. Ordering by timestamp alone would then make temporal
supersession nondeterministic; the monotonic `message_id` restores the true
order.

### 3.2 Conflict resolution by relation cardinality

The core temporal rule lives in `ContextEngineer._resolve_conflicts`. Facts are
grouped and the newest per group is kept, but *how* facts are grouped depends on
whether the relation is single- or multi-valued:

- **Single-valued (`EXCLUSIVE_RELATIONS = {WORKING_ON}`)** — grouped by
  `(subject, relation)`. A newer target **supersedes** the older one, so
  "working on REST" followed by "switch to GraphQL" yields only *GraphQL*.
- **Multi-valued (everything else)** — grouped by `(subject, relation, object)`.
  Distinct targets **coexist**; only exact duplicates dedupe. The user can
  `USES Redis` *and* `USES PostgreSQL` simultaneously.

This mirrors the notion of *functional properties* in RDF and *valid-time* in
temporal databases, applied at the granularity of individual relation types.

### 3.3 Suppressing dropped entities

When a message says "switch to GraphQL **instead of** REST", the phrase after
"instead of" names a technology being *abandoned*. Re-recording it would create a
fresh REST edge and defeat supersession, so the extractor strips
`instead of <word>` before scanning for technologies.

---

## 4. Extraction and Retrieval

### 4.1 Pattern-based extraction (default)

`extract_entities_simple` detects technologies by matching each message against a
19-item vocabulary (`TECH_VOCABULARY`) with word-boundary regexes that preserve
canonical casing. The relationship attached to those technologies is chosen by
**verb precedence**: preference verbs (*prefer/like/favorite*) →
`HAS_PREFERENCE`, beats project verbs (*build/working on/switch to*) →
`WORKING_ON`, beats generic use (*need/use/require*) → `USES`. Budget
(`Budget_$5000`), deadline (`Deadline_end_of_December`), and throughput
(`Performance_1000_requests`) constraints are pulled by dedicated patterns. The
design goal was to emit **only meaningful nodes** — an earlier naive version
produced junk nodes such as `using`, `prefer using`, and `should`.

### 4.2 Concept-expansion retrieval

`query_facts` tokenizes the query on non-alphanumerics (so `WORKING_ON` →
`{working, on}`) and expands two dictionaries before matching:

- **`TECH_CATEGORIES`** maps category words to concrete entities:
  `database → {PostgreSQL, MySQL, MongoDB, SQLite, Neo4j}`, `caching → {Redis}`,
  `stack → all`. This is what lets *"what database am I using?"* retrieve
  PostgreSQL although the query never names it.
- **`RELATION_INTENTS`** maps intent words to relation types:
  `requirements → HAS_CONSTRAINT`, `prefer → HAS_PREFERENCE`,
  `working → WORKING_ON`.

An edge matches if its tokens overlap the query, **or** its relation is a target
relation, **or** its object is a target technology. This is a hand-built,
dictionary-driven stand-in for semantic retrieval — not embeddings — and its
reach is exactly the size of those dictionaries.

### 4.3 Optional LLM extraction

With an OpenRouter API key (`--use-llm`), extraction is delegated to a free
instruction model (`openai/gpt-oss-20b`) prompted to return strict JSON, with
automatic fallback to pattern extraction on any error. As Section 5.3 shows, the
LLM path lifts the vocabulary and mixed-intent limits but does **not** solve
negation.

---

## 5. Evaluation

We characterize behavior on a fixed six-message reference dialogue (the built-in
`demo`): building a REST API; a \$5000 budget and end-of-December deadline; a
Python + FastAPI preference; a switch to GraphQL; adding Redis and PostgreSQL;
and a 1000-requests/second target. All results below are from direct measurement
(`test_battery.py`, 16/16 checks behaving as characterized). **These are
behavioral characterizations, not benchmark scores against baselines** — see
Section 7.

### 5.1 Graph construction

The six messages produce **11 nodes and 10 edges**: 3 `WORKING_ON`
(REST/API/GraphQL), 3 `HAS_CONSTRAINT`, 2 `HAS_PREFERENCE` (Python/FastAPI), and
2 `USES` (Redis/PostgreSQL). No junk nodes are emitted.

### 5.2 Correct behaviors

| Capability | Query | Result |
|---|---|---|
| Concept expansion (DB) | "what database am I using" | returns **PostgreSQL** |
| Concept expansion (cache) | "what am I using for caching" | returns **Redis** |
| Concept expansion (lang) | "what language do I prefer" | returns **Python** |
| Temporal supersession | "what am I working on" | **GraphQL only** (REST/API dropped) |
| Multi-valued coexistence | "what technologies do I use" | **both Redis and PostgreSQL** |
| Constraint extraction | "what are my constraints" | Budget + Deadline + Performance |

The end-to-end LLM path (Layer 6) produces answers **grounded in these retrieved
facts** rather than the model's general knowledge — e.g. it reports GraphQL as
the current project (correctly reflecting supersession) and enumerates the three
constraints the context layer selected.

### 5.3 Documented failure modes

Four probes deliberately expose the system's boundaries:

| Failure mode | Input | Pattern path | LLM path |
|---|---|---|---|
| **Out-of-vocabulary** | "backend in Rust with Axum" | **misses entirely** (0 edges) | captures Rust + Axum |
| **Negation** | "I do **not** want MongoDB" | records `USES MongoDB` (wrong) | records `HAS_PREFERENCE MongoDB` (**also wrong**) |
| **Mixed intent** | "I prefer Python but working on Django" | collapses to **one** relation | captures both |
| **No persistence / multi-user** | — | in-memory only; single hardcoded subject | same |

Negation is the most consequential: **neither** path handles it, so a stated
*rejection* of a technology is silently stored as an endorsement. The
vocabulary/mixed-intent limits are artifacts of the regex path that the LLM path
largely removes.

### 5.4 Known imprecision

Because `using` maps to the `USES` intent, a query like "what **database** am I
using" expands on both the `database` category *and* the `USES` relation, so it
can also surface `USES Redis` alongside PostgreSQL. Node-level results
disambiguate in practice, but retrieval favors recall over precision by design.

---

## 6. Limitations

1. **Fixed vocabulary.** Pattern extraction only sees the 19 listed technologies;
   anything else is invisible without the LLM path.
2. **No negation or hypotheticals.** "not", "don't", "instead of using" (beyond
   the one handled clause) are not modeled — a correctness risk, not just recall.
3. **Dictionary retrieval, not semantics.** Concept expansion covers exactly the
   hand-authored category/intent maps; it does not generalize (no embeddings).
4. **Single subject, in-memory only.** One hardcoded user; the graph is lost on
   exit — no persistence, no coreference, no multi-party dialogue.
5. **Coarse, self-supplied time.** All facts are timestamped at ingest; there is
   no extraction of *when* a fact is true in the world ("the deadline is in
   December" is stored, but December is not parsed into `event_time`).
6. **Uniform confidence.** Every edge is 0.9; the field exists but carries no
   signal.

---

## 7. Positioning and Related Work

The building blocks here are established, not novel. Retrieval-augmented
generation (Lewis et al., 2020) established external memory for LLMs; **MemGPT/
Letta** (Packer et al., 2023) framed context as a managed memory hierarchy;
**Zep/Graphiti**, **Mem0**, and **Microsoft GraphRAG** (2024) specifically build
*graph-structured*, and in Graphiti's case *bi-temporal*, agent memory with
extraction, retrieval, and conflict handling far beyond what is implemented here.
Temporal knowledge graphs are themselves a long-standing research area, and the
supersession rule in Section 3.2 is a restatement of RDF functional properties /
valid-time semantics.

What this project *is*: a clear, honest, end-to-end **reference implementation**
and teaching artifact — the full pipeline in a few hundred lines, every layer
observable, no hidden magic. That has real value as a portfolio piece,
course project, or explanatory scaffold. What it is *not*: a novel contribution
with an empirical claim.

**Could it become research-worthy?** Only with work it does not yet contain:

- A **novel claim** to test — e.g. that *relation-cardinality-aware* conflict
  resolution (single- vs multi-valued handled differently) reduces contradiction
  errors versus a uniform "keep-latest" policy; or that the coarse-clock
  `message_id` tiebreaker measurably improves supersession accuracy.
- An **evaluation against baselines** on an existing memory benchmark such as
  **LoCoMo** (Maharana et al., 2024) or **LongMemEval** (2024), reporting
  precision/recall of retrieved facts and answer accuracy versus a
  flat-RAG memory and versus Mem0/Zep.
- **Ablations** isolating each mechanism (concept expansion, cardinality rule,
  tiebreaker) so any gain is attributable.

Absent that, the intellectually interesting kernel — treating temporal conflict
resolution as a function of relation cardinality — remains a sound engineering
choice rather than a demonstrated result.

---

## 8. Conclusion

We described a compact temporal-knowledge-graph memory for LLM conversation:
timestamped facts, concept-expansion retrieval, and cardinality-aware temporal
conflict resolution, assembled into a grounded prompt through an observable
six-layer pipeline. Measured on a reference dialogue, it answers category
questions without literal overlap, respects temporal supersession, and preserves
coexisting facts — while transparently failing on out-of-vocabulary entities and,
in both extraction paths, on negation. The result is a strong systems
demonstration and teaching implementation whose ideas are well-grounded but not
new; turning its one genuinely thoughtful mechanism into a research contribution
would require a benchmark, baselines, and ablations that are currently absent.

---

## Appendix A. Reproducing the results

```bash
# Pattern path — full pipeline, no API key
printf 'demo\nwhat database am I using\nexit\n' | python tkg_cli.py

# LLM path — Layer 6 real generation (needs a free OpenRouter key)
python tkg_cli.py --use-llm --api-key sk-or-...

# Characterization battery (16 checks incl. failure probes)
python test_battery.py
```

## References (for positioning; not exhaustive)

- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* 2020.
- Packer et al. *MemGPT: Towards LLMs as Operating Systems.* 2023. (now Letta)
- Rasmussen et al. *Zep / Graphiti: temporal knowledge-graph agent memory.* 2024.
- *Mem0: memory layer for AI agents.* 2024.
- Microsoft Research. *GraphRAG.* 2024.
- Maharana et al. *LoCoMo: evaluating very long-term conversational memory.* 2024.
- *LongMemEval: benchmarking long-term interactive memory.* 2024.
- Hagberg, Schult, Swart. *NetworkX.* 2008.
