"""
Test battery for the TKG memory system.
Characterizes both correct behavior and known limitations.
No API key needed (pattern-based path).
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.getcwd())

from tkg_core import TKGMemory, create_demo_conversation
from context_engineer import ContextEngineer

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, cond, detail=""):
    results.append((name, PASS if cond else FAIL, detail))
    mark = "OK " if cond else "XX "
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))

def fresh_demo():
    tkg = TKGMemory()
    for m in create_demo_conversation():
        tkg.ingest_message(m, use_llm=False)
    return tkg

def answer_targets(tkg, query):
    """Return the resolved fact targets for a query, after context engineering."""
    ce = ContextEngineer(tkg)
    facts = tkg.query_facts(query)
    res = ce.build_system_prompt(query, facts)
    rels = [(f['relation'], f['to']) for f in res['facts_used'] if f['type'] == 'relationship']
    return rels

print("\n=== 1. GRAPH CONSTRUCTION (demo conversation) ===")
tkg = fresh_demo()
stats = tkg.get_graph_stats()
print("  stats:", stats)
check("6 messages processed", stats['messages_processed'] == 6)
check("edges are the 3 relation types", set(stats['relationship_types']) == {"WORKING_ON","HAS_CONSTRAINT","HAS_PREFERENCE","USES"}, str(stats['relationship_types']))

print("\n=== 2. CONCEPT-EXPANSION RETRIEVAL ===")
db = answer_targets(tkg, "what database am I using")
check("database -> PostgreSQL", ("USES","PostgreSQL") in db, str(db))
cache = answer_targets(tkg, "what am I using for caching")
check("caching -> Redis", ("USES","Redis") in cache, str(cache))
lang = answer_targets(tkg, "what language do I prefer")
check("language -> Python", ("HAS_PREFERENCE","Python") in lang, str(lang))

print("\n=== 3. TEMPORAL SUPERSESSION (single-valued) ===")
work = answer_targets(tkg, "what am I working on")
work_targets = [t for r,t in work if r == "WORKING_ON"]
check("WORKING_ON == GraphQL only (REST/API superseded)", work_targets == ["GraphQL"], str(work_targets))

print("\n=== 4. MULTI-VALUED RELATION COEXISTENCE ===")
uses = answer_targets(tkg, "what technologies do I use")
uses_targets = sorted(t for r,t in uses if r == "USES")
check("USES keeps BOTH Redis and PostgreSQL", "Redis" in uses_targets and "PostgreSQL" in uses_targets, str(uses_targets))

print("\n=== 5. CONSTRAINT EXTRACTION ===")
cons = answer_targets(tkg, "what are my constraints and requirements")
cons_targets = sorted(t for r,t in cons if r == "HAS_CONSTRAINT")
check("budget extracted", any("Budget" in t for t in cons_targets), str(cons_targets))
check("deadline extracted", any("Deadline" in t for t in cons_targets), str(cons_targets))
check("performance extracted", any("Performance" in t for t in cons_targets), str(cons_targets))

print("\n=== 6. LIMITATION PROBES (documenting real behavior) ===")

# 6a. Out-of-vocabulary technology
t = TKGMemory(); t.ingest_message("I'm building a backend in Rust with the Axum framework", use_llm=False)
rust = [ (u,v,d['type']) for u,v,d in t.graph.edges(data=True) ]
check("OOV tech (Rust/Axum) NOT captured -> vocabulary limit", len(rust) == 0, f"edges={rust}")

# 6b. Negation not handled
t = TKGMemory(); t.ingest_message("I definitely do not want to use MongoDB", use_llm=False)
neg = [ (u,v,d['type']) for u,v,d in t.graph.edges(data=True) ]
check("Negation NOT handled -> MongoDB wrongly recorded as USES", any(v=="MongoDB" for u,v,dt in neg), f"edges={neg}")

# 6c. Single relation type per message (verb precedence collapses mixed intents)
t = TKGMemory(); t.ingest_message("I prefer Python but I'm working on a Django migration", use_llm=False)
mixed = sorted({ d['type'] for u,v,d in t.graph.edges(data=True) })
check("Mixed-intent sentence collapses to ONE relation type", len(mixed) == 1, f"types={mixed}")

# 6d. No coreference / multi-user (hardcoded user_demo)
t = TKGMemory()
check("Single hardcoded subject (user_demo) only", t.user_id == "user_demo")

# 6e. Persistence
t = fresh_demo()
has_persist = hasattr(t, "save") or hasattr(t, "load")
check("No persistence layer (in-memory only)", not has_persist)

print("\n=== 7. RETRIEVAL PRECISION NOTE (over-recall) ===")
# 'database' intent also fires USES relation intent via 'using', so caching may leak in.
db_all = answer_targets(tkg, "what database am I using")
db_all_targets = sorted(f"{r}:{t}" for r,t in db_all)
print("  database-query returned:", db_all_targets)
check("database query includes PostgreSQL (may also over-recall Redis)", any("PostgreSQL" in x for x in db_all_targets), str(db_all_targets))

print("\n=== SUMMARY ===")
npass = sum(1 for _,s,_ in results if s == PASS)
print(f"  {npass}/{len(results)} checks behaved as characterized")
