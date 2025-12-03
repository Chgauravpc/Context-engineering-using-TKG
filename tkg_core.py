import networkx as nx
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import re


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
        Simplified entity extraction using pattern matching
        For demo - in production would use NER + trained models
        """
        entities = []
        relationships = []
        
        # Pattern matching for common entities
        patterns = {
            'technology': r'\b(Python|JavaScript|React|Node|FastAPI|Neo4j|TKG|GraphQL|REST|API|Redis|PostgreSQL|MongoDB|Docker|Kubernetes)\b',
            'task': r'\b(build|create|develop|implement|design|deploy|optimize)\s+(\w+)',
            'constraint': r'\b(budget|deadline|requirement|must|should)\b',
            'preference': r'\b(prefer|like|want|need)\s+(\w+)',
        }
        
        # Extract entities
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(0) if entity_type != 'task' else match.group(2)
                entities.append({
                    "name": entity_name.strip(),
                    "type": entity_type.title(),
                    "properties": {}
                })
        
        # Extract relationships from patterns
        if re.search(r'\b(working on|building|creating|developing)\b', message, re.IGNORECASE):
            tech_matches = re.findall(r'\b(Python|JavaScript|React|GraphQL|REST|API|app|project|system)\b', message, re.IGNORECASE)
            for tech in tech_matches:
                relationships.append({
                    "from": self.user_id,
                    "to": tech,
                    "type": "WORKING_ON"
                })
        
        if re.search(r'\b(budget|cost|spend)\b.*?(\$\d+k?|\d+\s*dollars)', message, re.IGNORECASE):
            budget_match = re.search(r'(\$\d+k?|\d+\s*dollars)', message, re.IGNORECASE)
            if budget_match:
                relationships.append({
                    "from": self.user_id,
                    "to": f"Budget_{budget_match.group(1)}",
                    "type": "HAS_CONSTRAINT"
                })
        
        if re.search(r'\b(prefer|like|want)\b', message, re.IGNORECASE):
            pref_match = re.search(r'(prefer|like|want)\s+(\w+)', message, re.IGNORECASE)
            if pref_match:
                relationships.append({
                    "from": self.user_id,
                    "to": pref_match.group(2),
                    "type": "HAS_PREFERENCE"
                })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
    
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
            
            # Using OpenRouter with free model (Meta Llama 3.1 8B Instruct)
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
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
        
        # Extract keywords from query
        keywords = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        # Search nodes
        for node_id, node_data in self.graph.nodes(data=True):
            if node_id == self.user_id:
                continue
            
            node_keywords = set(re.findall(r'\b\w+\b', node_id.lower()))
            if keywords & node_keywords:  # Intersection
                relevant_facts.append({
                    "type": "node",
                    "entity": node_id,
                    "entity_type": node_data.get("type"),
                    "first_mentioned": node_data.get("first_mentioned"),
                    "last_mentioned": node_data.get("last_mentioned")
                })
        
        # Search edges/relationships
        for from_node, to_node, edge_data in self.graph.edges(data=True):
            edge_text = f"{from_node} {edge_data.get('type', '')} {to_node}".lower()
            if any(kw in edge_text for kw in keywords):
                relevant_facts.append({
                    "type": "relationship",
                    "from": from_node,
                    "relation": edge_data.get("type"),
                    "to": to_node,
                    "event_time": edge_data.get("event_time"),
                    "confidence": edge_data.get("confidence", 0.9)
                })
        
        # Sort by recency (event_time)
        relevant_facts.sort(
            key=lambda x: x.get("event_time", x.get("last_mentioned", "")),
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