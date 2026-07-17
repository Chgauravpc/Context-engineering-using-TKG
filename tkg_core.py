import networkx as nx
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import re


# Known technologies, used both to extract clean entities and to answer
# category questions during retrieval.
TECH_VOCABULARY = [
    "Python", "JavaScript", "TypeScript", "React", "Node", "FastAPI", "Flask",
    "Django", "Neo4j", "GraphQL", "REST", "Redis", "PostgreSQL", "MySQL",
    "MongoDB", "SQLite", "Docker", "Kubernetes", "API",
]

# Concept words -> the technologies they refer to. Lets retrieval answer
# "what database am I using?" without the query literally naming PostgreSQL.
# Keys are matched against query tokens (both singular and plural forms listed).
TECH_CATEGORIES = {
    "database": {"PostgreSQL", "MySQL", "MongoDB", "SQLite", "Neo4j"},
    "databases": {"PostgreSQL", "MySQL", "MongoDB", "SQLite", "Neo4j"},
    "db": {"PostgreSQL", "MySQL", "MongoDB", "SQLite", "Neo4j"},
    "datastore": {"PostgreSQL", "MySQL", "MongoDB", "SQLite", "Neo4j", "Redis"},
    "cache": {"Redis"},
    "caching": {"Redis"},
    "language": {"Python", "JavaScript", "TypeScript"},
    "languages": {"Python", "JavaScript", "TypeScript"},
    "framework": {"FastAPI", "Flask", "Django", "React", "Node"},
    "frameworks": {"FastAPI", "Flask", "Django", "React", "Node"},
    "stack": set(TECH_VOCABULARY),
    "tech": set(TECH_VOCABULARY),
    "technology": set(TECH_VOCABULARY),
    "technologies": set(TECH_VOCABULARY),
}

# Query words -> relationship type they ask about. Lets retrieval return all
# WORKING_ON edges for "what am I working on?" etc.
RELATION_INTENTS = {
    "working": "WORKING_ON", "work": "WORKING_ON", "task": "WORKING_ON",
    "tasks": "WORKING_ON", "project": "WORKING_ON", "projects": "WORKING_ON",
    "building": "WORKING_ON", "build": "WORKING_ON",
    "constraint": "HAS_CONSTRAINT", "constraints": "HAS_CONSTRAINT",
    "requirement": "HAS_CONSTRAINT", "requirements": "HAS_CONSTRAINT",
    "budget": "HAS_CONSTRAINT", "deadline": "HAS_CONSTRAINT",
    "limit": "HAS_CONSTRAINT", "limits": "HAS_CONSTRAINT",
    "preference": "HAS_PREFERENCE", "preferences": "HAS_PREFERENCE",
    "prefer": "HAS_PREFERENCE", "prefers": "HAS_PREFERENCE",
    "like": "HAS_PREFERENCE", "likes": "HAS_PREFERENCE",
    "using": "USES", "uses": "USES", "use": "USES",
    "need": "USES", "needs": "USES", "dependencies": "USES",
}


class TKGMemory:
    """
    Temporal Knowledge Graph Memory System
    Stores conversation facts as time-stamped graph with entities and relations
    """
    
    def __init__(self, openrouter_api_key: Optional[str] = None):
        """Initialize TKG with empty graph"""
        self.graph = nx.MultiDiGraph()
        self.message_count = 0
        self.api_key = openrouter_api_key
        self.user_id = "user_demo"
        
        # Add user node
        self.graph.add_node(
            self.user_id,
            type="User",
            created_at=datetime.now().isoformat()
        )
    
    def extract_entities_simple(self, message: str) -> Dict[str, Any]:
        """
        Pattern-based entity/relationship extraction (no LLM).

        Strategy: detect known technologies against a vocabulary and attach
        them to the relationship the sentence implies (preference / working-on /
        uses), then pull out budget, deadline and performance constraints. This
        avoids the noisy trigger-word nodes the naive version produced (e.g.
        "using", "prefer using", "should").
        """
        entities = []
        relationships = []
        msg_lower = message.lower()

        def add_entity(name, etype):
            entities.append({"name": name, "type": etype, "properties": {}})

        def add_rel(target, rel_type):
            relationships.append({
                "from": self.user_id, "to": target, "type": rel_type
            })

        # --- Technologies ---------------------------------------------------
        # Ignore a technology named in an "instead of X" clause: it is being
        # dropped, so it must not be re-recorded (which would defeat the
        # temporal supersession, e.g. "switch to GraphQL instead of REST").
        scan = re.sub(r'\binstead of\s+\w+', '', message, flags=re.IGNORECASE)

        found_tech = []
        for tech in TECH_VOCABULARY:
            if re.search(rf'\b{re.escape(tech)}\b', scan, re.IGNORECASE):
                if tech not in found_tech:
                    found_tech.append(tech)  # canonical casing

        # Decide the relationship these technologies have with the user, from
        # the verbs in the sentence. Order matters: preference beats working-on
        # beats generic use ("I prefer using Python" is a preference).
        if re.search(r'\b(prefer|prefers|like|likes|favou?rite|favou?rites)\b', msg_lower):
            tech_rel = "HAS_PREFERENCE"
        elif re.search(r'\b(working on|build|building|creating|create|developing|develop)\b'
                       r'|\b(switch(?:ing)?|migrat(?:e|ing)|mov(?:e|ing))\s+to\b', msg_lower):
            tech_rel = "WORKING_ON"
        elif re.search(r'\b(need|needs|use|uses|using|require|requires|add|adding|integrate)\b', msg_lower):
            tech_rel = "USES"
        else:
            tech_rel = None

        for tech in found_tech:
            add_entity(tech, "Technology")
            if tech_rel:
                add_rel(tech, tech_rel)

        # --- Budget constraint ---------------------------------------------
        if re.search(r'\b(budget|cost|spend|price|pricing)\b', msg_lower):
            budget_match = re.search(r'(\$\s?\d[\d,]*\s?k?|\d[\d,]*\s*dollars)', message, re.IGNORECASE)
            if budget_match:
                amount = re.sub(r'\s+', '', budget_match.group(1))
                name = f"Budget_{amount}"
                add_entity(name, "Constraint")
                add_rel(name, "HAS_CONSTRAINT")

        # --- Deadline constraint -------------------------------------------
        deadline_match = re.search(r'\bdeadline\b\s*(?:is|:)?\s*([\w ]+?)(?:[.,;]|$)', message, re.IGNORECASE)
        if deadline_match:
            phrase = deadline_match.group(1).strip()
            if phrase:
                name = "Deadline_" + re.sub(r'\s+', '_', phrase)
                add_entity(name, "Constraint")
                add_rel(name, "HAS_CONSTRAINT")

        # --- Performance/throughput constraint -----------------------------
        perf_match = re.search(r'(\d[\d,]*)\s*(requests?|rps|qps|users?|connections?)\b', msg_lower)
        if perf_match:
            name = f"Performance_{perf_match.group(1)}_{perf_match.group(2)}"
            add_entity(name, "Constraint")
            add_rel(name, "HAS_CONSTRAINT")

        return {"entities": entities, "relationships": relationships}
    
    def extract_entities_llm(self, message: str) -> Dict[str, Any]:
        """
        Extract entities using LLM (OpenRouter with free models)
        Fallback to simple extraction if no API key
        """
        if not self.api_key:
            return self.extract_entities_simple(message)
        
        try:
            import requests
            
            prompt = f"""Extract entities and relationships from this message. Return ONLY valid JSON.

Message: "{message}"

Return format:
{{
  "entities": [
    {{"name": "entity_name", "type": "Person|Technology|Task|Constraint|Preference|Concept", "properties": {{}}}}
  ],
  "relationships": [
    {{"from": "entity_or_user", "to": "entity", "type": "WORKING_ON|NEEDS|USES|HAS_CONSTRAINT|HAS_PREFERENCE|DECIDED"}}
  ]
}}

Rules:
- Extract concrete entities (technologies, tasks, constraints, preferences)
- Use "user_demo" as the from entity when user is the subject
- Keep entity names simple and clear
- Return ONLY JSON, no markdown or extra text"""
            
            # Using OpenRouter with a currently-free model (OpenAI gpt-oss-20b).
            # The old llama-3.1-8b:free tier was moved to paid-only upstream.
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-20b:free",
                    "messages": [
                        {"role": "system", "content": "You extract entities and return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"⚠️  OpenRouter API error: {response.status_code}")
                if response.status_code == 401:
                    print("    Invalid API key. Get one from: https://openrouter.ai/keys")
                return self.extract_entities_simple(message)
            
            result = response.json()["choices"][0]["message"]["content"].strip()
            
            # Clean markdown formatting if present
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:].strip()
            
            return json.loads(result)
        
        except requests.exceptions.Timeout:
            print(f"⚠️  OpenRouter API timeout, using simple extraction")
            return self.extract_entities_simple(message)
        except requests.exceptions.RequestException as e:
            print(f"⚠️  OpenRouter API connection error: {e}")
            return self.extract_entities_simple(message)
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse LLM response as JSON")
            return self.extract_entities_simple(message)
        except Exception as e:
            print(f"⚠️  LLM extraction failed: {e}, using simple extraction")
            return self.extract_entities_simple(message)
    
    def ingest_message(self, message: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Main ingestion pipeline: message -> facts -> graph
        Returns extraction result for transparency
        """
        timestamp = datetime.now().isoformat()
        self.message_count += 1
        
        # Step 1: Extract entities and relations
        if use_llm:
            extracted = self.extract_entities_llm(message)
        else:
            extracted = self.extract_entities_simple(message)
        
        # Step 2: Add to graph with temporal properties
        for entity in extracted.get("entities", []):
            node_id = entity["name"]
            
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    type=entity.get("type", "Entity"),
                    properties=entity.get("properties", {}),
                    first_mentioned=timestamp,
                    message_id=self.message_count
                )
            
            # Update last_mentioned
            self.graph.nodes[node_id]["last_mentioned"] = timestamp
        
        # Step 3: Add relationships with temporal edges
        for rel in extracted.get("relationships", []):
            self.graph.add_edge(
                rel["from"],
                rel["to"],
                type=rel["type"],
                event_time=timestamp,
                ingest_time=timestamp,
                message_id=self.message_count,
                confidence=0.9,
                active=True
            )
        
        return {
            "message_id": self.message_count,
            "extracted": extracted,
            "timestamp": timestamp
        }
    
    def query_facts(self, query_text: str, time_window_hours: Optional[int] = None) -> List[Dict]:
        """
        Query the TKG for relevant facts using Python graph traversal
        Returns list of facts with temporal info and confidence
        """
        relevant_facts = []

        # Extract keywords from query. Split on non-alphanumerics so that
        # underscores in stored names/relations (e.g. WORKING_ON) tokenize into
        # separate words and stay matchable.
        keywords = set(re.findall(r'[a-z0-9]+', query_text.lower()))

        # Concept expansion: turn category words ("database", "cache", "stack")
        # into the concrete technologies they mean, and intent words ("working",
        # "requirements", "prefer") into the relationship types they ask about.
        # This is what lets "what database am I using?" find PostgreSQL even
        # though the query never says "PostgreSQL".
        target_techs = set()
        target_relations = set()
        for kw in keywords:
            if kw in TECH_CATEGORIES:
                target_techs |= TECH_CATEGORIES[kw]
            if kw in RELATION_INTENTS:
                target_relations.add(RELATION_INTENTS[kw])
        target_techs_lower = {t.lower() for t in target_techs}

        # Search nodes
        for node_id, node_data in self.graph.nodes(data=True):
            if node_id == self.user_id:
                continue

            node_keywords = set(re.findall(r'[a-z0-9]+', node_id.lower()))
            if (keywords & node_keywords) or (node_id.lower() in target_techs_lower):
                relevant_facts.append({
                    "type": "node",
                    "entity": node_id,
                    "entity_type": node_data.get("type"),
                    "first_mentioned": node_data.get("first_mentioned"),
                    "last_mentioned": node_data.get("last_mentioned"),
                    "message_id": node_data.get("message_id", 0)
                })

        # Search edges/relationships
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get("type", "")
            edge_text = f"{from_node} {rel_type} {to_node}".lower()
            # Match on whole-word tokens (not substrings) so short query words
            # like "i" don't spuriously match unrelated edges (e.g. "i" inside
            # "constraint"). Underscores split so WORKING_ON -> {working, on}.
            edge_keywords = set(re.findall(r'[a-z0-9]+', edge_text))
            if (keywords & edge_keywords
                    or rel_type in target_relations
                    or to_node.lower() in target_techs_lower):
                relevant_facts.append({
                    "type": "relationship",
                    "from": from_node,
                    "relation": rel_type,
                    "to": to_node,
                    "event_time": edge_data.get("event_time"),
                    "confidence": edge_data.get("confidence", 0.9),
                    "message_id": edge_data.get("message_id", 0)
                })

        # Sort by recency (event_time), then message_id as a tiebreaker. The
        # tiebreaker matters because datetime.now() has coarse resolution on
        # some platforms (~15ms on Windows), so messages ingested in a tight
        # loop can share a timestamp; message_id preserves their true order.
        # Coerce missing/None timestamps to "" so facts without a timestamp
        # (e.g. auto-created edge-target nodes) don't break the comparison.
        relevant_facts.sort(
            key=lambda x: (x.get("event_time") or x.get("last_mentioned") or "",
                           x.get("message_id", 0)),
            reverse=True
        )
        
        return relevant_facts
    
    def get_context_summary(self, max_facts: int = 8) -> str:
        """
        Generate context summary for LLM prompt
        Returns formatted evidence block
        """
        recent_edges = []
        
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            recent_edges.append({
                "from": from_node,
                "rel": edge_data.get("type"),
                "to": to_node,
                "time": edge_data.get("event_time"),
                "conf": edge_data.get("confidence", 0.9)
            })
        
        # Sort by time
        recent_edges.sort(key=lambda x: x["time"], reverse=True)
        recent_edges = recent_edges[:max_facts]
        
        # Format as evidence block
        evidence_lines = ["EVIDENCE (from conversation memory):"]
        for edge in recent_edges:
            time_short = edge["time"][:19] if edge["time"] else "unknown"
            evidence_lines.append(
                f"- [{time_short}] {edge['from']} {edge['rel']} {edge['to']} (confidence={edge['conf']:.2f})"
            )
        
        return "\n".join(evidence_lines)
    
    def get_graph_stats(self) -> Dict:
        """Return graph statistics for dashboard"""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "messages_processed": self.message_count,
            "node_types": self._count_node_types(),
            "relationship_types": self._count_relationship_types()
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type"""
        type_counts = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_relationship_types(self) -> Dict[str, int]:
        """Count edges by relationship type"""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "Unknown")
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        return type_counts
    
    def export_graph(self) -> Dict:
        """Export graph as JSON for visualization"""
        nodes = []
        edges = []
        
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": node_id,
                "type": data.get("type", "Unknown"),
                "title": f"Type: {data.get('type')}\nFirst: {data.get('first_mentioned', 'N/A')[:19]}"
            })
        
        for from_node, to_node, data in self.graph.edges(data=True):
            edges.append({
                "from": from_node,
                "to": to_node,
                "label": data.get("type", ""),
                "title": f"{data.get('type')}\nTime: {data.get('event_time', 'N/A')[:19]}",
                "arrows": "to"
            })
        
        return {"nodes": nodes, "edges": edges}


# Helper function for demo scenarios
def create_demo_conversation() -> List[str]:
    """Sample conversation for demo"""
    return [
        "I'm building a REST API for my e-commerce platform",
        "The budget is $5000 and deadline is end of December",
        "I prefer using Python with FastAPI framework",
        "Actually, let's switch to GraphQL instead of REST",
        "We also need Redis for caching and PostgreSQL for the database",
        "The project should handle 1000 requests per second"
    ]