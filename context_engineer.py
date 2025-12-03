from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict


class ContextEngineer:
    """
    Context Engineering Layer - Generates system prompts from TKG memory
    
    Responsibilities:
    1. Relevance filtering - select facts relevant to current query
    2. Temporal consistency - resolve conflicts using timestamps
    3. Context compression - keep only essential facts
    4. Prompt assembly - structure facts into coherent system prompt
    """
    
    def __init__(self, tkg_memory):
        self.tkg = tkg_memory
        self.max_facts = 10
        self.recency_boost_hours = 24
    
    def build_system_prompt(self, user_message: str, retrieved_facts: List[Dict]) -> Dict[str, Any]:
        """
        Main entry point: Build system prompt from TKG facts
        
        Returns:
            {
                'system_prompt': str,
                'stats': dict,
                'reasoning': list  # Why certain facts were included/excluded
            }
        """
        reasoning = []
        
        # Step 1: Filter relevant facts
        relevant_facts = self._filter_relevant(user_message, retrieved_facts, reasoning)
        reasoning.append(f"Filtered {len(retrieved_facts)} facts → {len(relevant_facts)} relevant")
        
        # Step 2: Resolve temporal conflicts
        resolved_facts = self._resolve_conflicts(relevant_facts, reasoning)
        reasoning.append(f"Resolved conflicts → {len(resolved_facts)} consistent facts")
        
        # Step 3: Compress to top N facts
        compressed_facts = self._compress_facts(resolved_facts, reasoning)
        reasoning.append(f"Compressed to top {len(compressed_facts)} facts")
        
        # Step 4: Assemble structured system prompt
        system_prompt = self._assemble_prompt(compressed_facts, user_message)
        
        return {
            'system_prompt': system_prompt,
            'facts_used': compressed_facts,
            'reasoning': reasoning,
            'stats': {
                'total_facts': len(retrieved_facts),
                'relevant_facts': len(relevant_facts),
                'final_facts': len(compressed_facts)
            }
        }
    
    def _filter_relevant(self, query: str, facts: List[Dict], reasoning: List[str]) -> List[Dict]:
        """
        Filter facts based on relevance to current query
        Uses keyword matching + recency boost
        """
        if not facts:
            return []
        
        query_keywords = set(query.lower().split())
        scored_facts = []
        
        for fact in facts:
            score = 0
            
            # Keyword matching
            if fact['type'] == 'relationship':
                fact_text = f"{fact['from']} {fact['relation']} {fact['to']}".lower()
            else:
                fact_text = f"{fact['entity']} {fact.get('entity_type', '')}".lower()
            
            # Score based on keyword overlap
            fact_words = set(fact_text.split())
            overlap = query_keywords & fact_words
            score += len(overlap) * 2
            
            # Recency boost (facts from last 24 hours get priority)
            event_time = fact.get('event_time') or fact.get('last_mentioned')
            if event_time:
                try:
                    fact_time = datetime.fromisoformat(event_time)
                    age_hours = (datetime.now() - fact_time).total_seconds() / 3600
                    if age_hours < self.recency_boost_hours:
                        score += 5
                        reasoning.append(f"Boosted recent fact: {fact_text[:50]}")
                except:
                    pass
            
            # Relationship facts are generally more informative
            if fact['type'] == 'relationship':
                score += 1
            
            if score > 0:
                scored_facts.append((score, fact))
        
        # Sort by score descending
        scored_facts.sort(reverse=True, key=lambda x: x[0])
        
        return [fact for score, fact in scored_facts]
    
    def _resolve_conflicts(self, facts: List[Dict], reasoning: List[str]) -> List[Dict]:
        """
        Resolve temporal conflicts - keep latest version of conflicting facts
        Example: If user changed preference from 'dark' to 'light', keep 'light'
        """
        # Group facts by entity or relationship signature
        fact_groups = defaultdict(list)
        
        for fact in facts:
            if fact['type'] == 'relationship':
                # Group by (from, relation) - allows tracking changes
                key = (fact['from'], fact['relation'])
                fact_groups[key].append(fact)
            else:
                # Group by entity name
                key = ('entity', fact['entity'])
                fact_groups[key].append(fact)
        
        resolved = []
        
        for key, group in fact_groups.items():
            if len(group) == 1:
                resolved.append(group[0])
            else:
                # Multiple facts for same entity/relationship - pick latest
                sorted_group = sorted(
                    group, 
                    key=lambda f: f.get('event_time', f.get('last_mentioned', '')),
                    reverse=True
                )
                
                latest = sorted_group[0]
                resolved.append(latest)
                
                # Note the conflict resolution
                if len(sorted_group) > 1:
                    old_value = sorted_group[1]
                    reasoning.append(
                        f"Conflict resolved: Using latest fact for {key[1]} "
                        f"(supersedes {len(sorted_group)-1} older facts)"
                    )
        
        return resolved
    
    def _compress_facts(self, facts: List[Dict], reasoning: List[str]) -> List[Dict]:
        """
        Compress to top N most important facts
        Priority: Recent relationships > Recent entities > Older facts
        """
        if len(facts) <= self.max_facts:
            return facts
        
        # Sort by importance
        def fact_importance(fact):
            score = 0
            
            # Relationships are more informative
            if fact['type'] == 'relationship':
                score += 10
            
            # Recent facts are more important
            event_time = fact.get('event_time') or fact.get('last_mentioned')
            if event_time:
                try:
                    fact_time = datetime.fromisoformat(event_time)
                    hours_old = (datetime.now() - fact_time).total_seconds() / 3600
                    # Decay score over time
                    score += max(0, 10 - (hours_old / 24))
                except:
                    pass
            
            # Certain relationship types are more important
            important_relations = ['WORKING_ON', 'HAS_CONSTRAINT', 'HAS_PREFERENCE', 'DECIDED']
            if fact.get('relation') in important_relations:
                score += 5
            
            return score
        
        sorted_facts = sorted(facts, key=fact_importance, reverse=True)
        compressed = sorted_facts[:self.max_facts]
        
        reasoning.append(f"Compressed: Kept top {self.max_facts} of {len(facts)} facts")
        
        return compressed
    
    def _assemble_prompt(self, facts: List[Dict], user_message: str) -> str:
        """
        Assemble final system prompt with structure and instructions
        """
        # Separate facts by category
        profile_facts = []
        task_facts = []
        preference_facts = []
        constraint_facts = []
        other_facts = []
        
        for fact in facts:
            if fact['type'] == 'relationship':
                rel_type = fact['relation']
                fact_str = f"- {fact['from']} {rel_type} {fact['to']}"
                
                if 'event_time' in fact:
                    time_str = fact['event_time'][:19]
                    fact_str += f" (as of {time_str})"
                
                if rel_type == 'WORKING_ON':
                    task_facts.append(fact_str)
                elif rel_type in ['HAS_PREFERENCE', 'PREFERS']:
                    preference_facts.append(fact_str)
                elif rel_type in ['HAS_CONSTRAINT', 'MUST', 'DEADLINE']:
                    constraint_facts.append(fact_str)
                else:
                    other_facts.append(fact_str)
            else:
                # Entity fact
                entity_str = f"- {fact['entity']} (type: {fact.get('entity_type', 'Unknown')})"
                profile_facts.append(entity_str)
        
        # Build structured prompt
        sections = []
        
        sections.append("### USER MEMORY CONTEXT")
        sections.append("You are an AI assistant with access to the user's conversation memory stored in a Temporal Knowledge Graph.")
        sections.append("")
        
        if task_facts:
            sections.append("**Current Tasks/Projects:**")
            sections.extend(task_facts)
            sections.append("")
        
        if constraint_facts:
            sections.append("**Constraints & Requirements:**")
            sections.extend(constraint_facts)
            sections.append("")
        
        if preference_facts:
            sections.append("**User Preferences:**")
            sections.extend(preference_facts)
            sections.append("")
        
        if other_facts:
            sections.append("**Other Relevant Facts:**")
            sections.extend(other_facts)
            sections.append("")
        
        if profile_facts:
            sections.append("**Entities Mentioned:**")
            sections.extend(profile_facts)
            sections.append("")
        
        # Add temporal notes
        sections.append("### TEMPORAL NOTES")
        recent_count = sum(1 for f in facts if self._is_recent(f, hours=24))
        sections.append(f"- {recent_count} of these facts were updated in the last 24 hours")
        sections.append(f"- All facts are timestamped and represent the latest known state")
        sections.append("")
        
        # Add instructions for LLM
        sections.append("### INSTRUCTIONS")
        sections.append("- **Ground your responses** in the facts provided above")
        sections.append("- **Reference ongoing projects** and tasks when relevant")
        sections.append("- **Maintain consistency** with stored preferences and constraints")
        sections.append("- **Use temporal awareness**: newer facts supersede older ones")
        sections.append("- If information is **missing** from memory, ask clarifying questions")
        sections.append("- **Do not contradict** memory facts unless user explicitly corrects them")
        sections.append("- Be **concise and technically accurate** in your responses")
        sections.append("")
        
        return "\n".join(sections)
    
    def _is_recent(self, fact: Dict, hours: int = 24) -> bool:
        """Check if fact is from last N hours"""
        event_time = fact.get('event_time') or fact.get('last_mentioned')
        if not event_time:
            return False
        
        try:
            fact_time = datetime.fromisoformat(event_time)
            return (datetime.now() - fact_time) < timedelta(hours=hours)
        except:
            return False
    
    def generate_user_prompt(self, user_message: str, system_prompt_result: Dict) -> str:
        """
        Generate the final user prompt that includes context
        """
        return f"""Based on the memory context above, please respond to:

USER: {user_message}"""


def format_system_prompt_display(result: Dict) -> str:
    """
    Format the system prompt result for nice CLI display
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("SYSTEM PROMPT (Generated by Context Engineering Layer)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(result['system_prompt'])
    lines.append("")
    lines.append("-" * 80)
    lines.append("CONTEXT ENGINEERING STATS")
    lines.append("-" * 80)
    stats = result['stats']
    lines.append(f"Total facts retrieved:  {stats['total_facts']}")
    lines.append(f"Relevant facts:         {stats['relevant_facts']}")
    lines.append(f"Facts in final prompt:  {stats['final_facts']}")
    lines.append("")
    lines.append("Reasoning:")
    for reason in result['reasoning']:
        lines.append(f"  • {reason}")
    lines.append("=" * 80)
    lines.append("")
    
    return "\n".join(lines)