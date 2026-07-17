# 🧠 TKG Memory System for LLMs

**Temporal Knowledge Graph memory with automatic context engineering**

A compact, fully-observable implementation of temporal-knowledge-graph memory for
LLM conversations. Facts extracted from your messages are stored as a timestamped
graph, retrieved by concept expansion, and engineered into a grounded system
prompt — with every layer of the pipeline printed so you can see exactly what the
model receives.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Scope:** this is a reference implementation and teaching artifact — a full
> memory pipeline in a few hundred readable lines. It is *not* a production
> memory service, and it has real, documented limitations (see
> [Limitations](#-limitations)). For a precise write-up of the design, measured
> behavior, and honest positioning against existing systems, see
> **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

---

## 🎯 What problem does this solve?

An LLM is stateless across turns. Without external memory it forgets facts you
stated moments ago, and — more subtly — it can't tell when a new statement
**contradicts** an old one. Plain RAG retrieves text chunks but has no notion of
*when* a fact became true, so "I'm building a REST API" and "let's switch to
GraphQL" both come back, and the model has to guess.

This system:

- Stores facts in a **temporal knowledge graph** (every edge carries an event time)
- **Resolves conflicts by relation cardinality** — a newer `WORKING_ON` supersedes
  the old one, while `USES Redis` and `USES PostgreSQL` correctly coexist
- Retrieves via **concept expansion** — "what database am I using?" finds
  PostgreSQL even though the query never says "PostgreSQL"
- Assembles the surviving facts into a **structured, grounded system prompt**

---

## 🏗️ Architecture

Six layers, executed every turn:

```
User message
   │
   ▼
[1] Entity / relation extraction   →  facts {entities, relationships}
   │
   ▼
[2] TKG write                      →  timestamped nodes + edges
   │
   ▼
[3] Retrieval (concept expansion)  →  candidate facts, recency-sorted
   │
   ▼
[4] Context engineering            →  relevance filter → conflict
   │                                   resolution → compression
   ▼
[5] System-prompt assembly         →  structured, sectioned prompt
   │
   ▼
[6] LLM response (optional)        →  grounded answer
```

Layers 1–3 own the graph (`tkg_core.py`). Layers 4–5 own prompt construction and
never mutate the graph (`context_engineer.py`). Layer 6 is a thin driver
(`tkg_cli.py`) that either calls an LLM or produces a deterministic simulated
answer — so the memory machinery is fully testable with **no API key**.

---

## 🚀 Quick start

```bash
git clone https://github.com/Chgauravpc/Context-engineering-using-TKG.git
cd Context-engineering-using-TKG
pip install -r requirements.txt
```

**Option 1 — no API key** (the whole pipeline except Layer 6 generation):

```bash
python tkg_cli.py

YOU: demo                      # loads the 6-message sample conversation
YOU: what am I working on      # watch context engineering resolve REST → GraphQL
YOU: stats                     # graph statistics
YOU: exit
```

**Option 2 — with a free LLM** (real extraction + generation):

```bash
# Free key, no credit card: https://openrouter.ai/keys
python tkg_cli.py --use-llm --api-key sk-or-v1-YOUR-KEY-HERE
```

**Run the test battery** (16 checks, including failure-mode probes):

```bash
python test_battery.py
```

---

## 💡 How it works

### Temporal conflict resolution by relation cardinality

The core rule (`ContextEngineer._resolve_conflicts`) groups facts differently
depending on whether a relation holds one value or many:

- **Single-valued** (`EXCLUSIVE_RELATIONS = {WORKING_ON}`) — grouped by
  `(subject, relation)`. A newer target **supersedes** the older one, so
  *"building a REST API"* → *"switch to GraphQL"* leaves only **GraphQL**.
- **Multi-valued** (everything else) — grouped by `(subject, relation, object)`.
  Distinct targets **coexist**, so you keep both **Redis** and **PostgreSQL**.

Edges also carry a monotonic `message_id` used as a **tiebreaker**: on Windows
`datetime.now()` has ~15 ms resolution, so messages ingested in a tight loop can
share a timestamp — ordering by time alone would make supersession
nondeterministic.

### Concept-expansion retrieval

Two dictionaries expand the query before matching:

- `TECH_CATEGORIES` — `database → {PostgreSQL, MySQL, MongoDB, SQLite, Neo4j}`,
  `caching → {Redis}`, `stack → all`
- `RELATION_INTENTS` — `requirements → HAS_CONSTRAINT`, `prefer → HAS_PREFERENCE`,
  `working → WORKING_ON`

This is a hand-built stand-in for semantic search — **not embeddings** — so its
reach is exactly the size of those dictionaries.

### Full transparency

Every run prints extraction results, retrieved facts, the context-engineering
reasoning trace, **the generated system prompt**, and the final response.

---

## 📊 Measured behavior

On the built-in 6-message `demo` conversation the graph is **11 nodes / 10 edges**
(3 `WORKING_ON`, 3 `HAS_CONSTRAINT`, 2 `HAS_PREFERENCE`, 2 `USES`) with no junk
nodes. Verified by `test_battery.py` (16/16 checks):

| Capability | Query | Result |
|---|---|---|
| Concept expansion (DB) | "what database am I using" | **PostgreSQL** |
| Concept expansion (cache) | "what am I using for caching" | **Redis** |
| Concept expansion (lang) | "what language do I prefer" | **Python** |
| Temporal supersession | "what am I working on" | **GraphQL only** (REST/API dropped) |
| Multi-valued coexistence | "what technologies do I use" | **Redis + PostgreSQL** |
| Constraint extraction | "what are my constraints" | Budget + Deadline + Performance |

With `--use-llm`, Layer 6 answers are grounded in exactly these retrieved facts —
it reports GraphQL as the current project (respecting supersession) and
enumerates the three constraints the context layer selected.

---

## ⚠️ Limitations

These are real and measured, not hypothetical:

| Limitation | Detail |
|---|---|
| **Fixed vocabulary** | Pattern extraction only recognizes 19 hard-coded technologies. "Rust with Axum" → **0 edges**. The `--use-llm` path fixes this. |
| **Negation is not handled** | "I do **not** want MongoDB" is stored as a *positive* fact. **Both** the pattern path and the LLM path get this wrong. |
| **Mixed intent** | One relation type per message on the pattern path (verb precedence). The LLM path handles this. |
| **Dictionary retrieval** | Concept expansion covers only the hand-authored maps; it does not generalize. |
| **No persistence** | In-memory only — the graph is lost on exit. |
| **Single user** | One hardcoded subject (`user_demo`); no coreference, no multi-party dialogue. |
| **Ingest-time only** | "the deadline is in December" is stored, but December is never parsed into `event_time`. |
| **Uniform confidence** | Every edge is `0.9`; the field carries no signal. |

Retrieval also favors **recall over precision**: because `using` maps to the
`USES` intent, "what *database* am I using" can surface Redis alongside
PostgreSQL.

---

## 📁 Project structure

```
├── tkg_core.py            # Layers 1-3: extraction, storage, retrieval
├── context_engineer.py    # Layers 4-5: context engineering & prompt assembly
├── tkg_cli.py             # Layer 6: CLI + LLM integration
├── test_battery.py        # 16 characterization checks incl. failure probes
├── TECHNICAL_REPORT.md    # Design, evaluation, related work, honest positioning
├── requirements.txt
├── LICENSE
└── Readme.md
```

---

## 🔧 Technical details

**Storage** — NetworkX in-memory `MultiDiGraph`; traversal via the Python API
(not Cypher). Suitable for demos; a production path would move to Neo4j.

**LLM** — [OpenRouter](https://openrouter.ai) free tier, `openai/gpt-oss-20b`,
with automatic fallback to pattern extraction on any API error. No OpenAI
dependency.

> **Note:** the model id is pinned in `tkg_core.py` and `tkg_cli.py`. OpenRouter
> moves models between free and paid tiers, so if you see
> `404 – this model is unavailable for free`, swap it for a currently-free model
> from https://openrouter.ai/models.

**Context engineering algorithm**
1. **Filter** — token overlap + recency boost (with a most-recent fallback if
   nothing scores)
2. **Resolve** — group by relation cardinality, keep latest (timestamp, then
   `message_id`)
3. **Compress** — score by importance, keep top `max_facts` (default 10)
4. **Assemble** — structure into sections with grounding instructions

---

## 📈 Roadmap

**Current**
- ✅ NetworkX in-memory storage with temporal edges
- ✅ Pattern-based + optional LLM extraction
- ✅ Concept-expansion retrieval
- ✅ Cardinality-aware conflict resolution
- ✅ Characterization test battery

**Next — correctness**
- [ ] **Negation handling** (highest-value fix — currently stores rejections as endorsements)
- [ ] Parse real-world times into `event_time` (bi-temporal model)
- [ ] Persistence (JSON snapshot, then Neo4j)

**Later**
- [ ] Vector embeddings for hybrid retrieval (replace dictionary expansion)
- [ ] Multi-user support and coreference
- [ ] Benchmark against LoCoMo / LongMemEval with baselines and ablations

---

## 🤝 Contributing

Contributions welcome — the [Limitations](#-limitations) table is the best place
to start, and negation handling is the highest-impact open issue.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

Built on [NetworkX](https://networkx.org/) for graph operations and
[OpenRouter](https://openrouter.ai) for free LLM access. The design draws on
established work in agent memory — MemGPT/Letta, Zep/Graphiti, Mem0, and
GraphRAG — discussed in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

---

**Author:** [@Chgauravpc](https://github.com/Chgauravpc)
