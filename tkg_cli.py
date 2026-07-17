#!/usr/bin/env python3
"""
Usage: python tkg_cli.py [--use-llm] [--api-key YOUR_KEY]
"""

import sys
import argparse
from tkg_core import TKGMemory, create_demo_conversation
from context_engineer import ContextEngineer, format_system_prompt_display
import requests
from typing import Dict


class TKGChatCLI:
    """CLI interface for TKG Memory System with full pipeline"""
    
    def __init__(self, use_llm=False, api_key=None):
        self.tkg = TKGMemory(openrouter_api_key=api_key)
        self.context_engineer = ContextEngineer(self.tkg)
        self.use_llm = use_llm
        self.api_key = api_key
        self.conversation_history = []
        
        # Test API key if provided
        if use_llm and api_key:
            self._test_api_key()
    
    def print_header(self):
        """Print welcome header"""
        print("\n" + "=" * 80)
        print("🧠 TEMPORAL KNOWLEDGE GRAPH MEMORY SYSTEM")
        print("=" * 80)
        print("\nArchitecture Layers Active:")
        print("  [1] NLU Extraction Layer      ✓")
        print("  [2] TKG Memory Write Layer    ✓")
        print("  [3] TKG Memory Retrieval      ✓")
        print("  [4] Context Engineering       ✓")
        print("  [5] System Prompt Generation  ✓")
        if self.use_llm and self.api_key:
            print("  [6] LLM Response Generation   ✓ (OpenRouter - gpt-oss-20b)")
        else:
            print("  [6] LLM Response Generation   ⊗ (Simulated - no API key)")
        print("\n" + "=" * 80)
        print("\nCommands:")
        print("  Type your message to chat")
        print("  'demo'  - Load demo conversation")
        print("  'stats' - Show graph statistics")
        print("  'query <text>' - Query specific facts")
        print("  'clear' - Clear conversation")
        print("  'exit'  - Quit")
        print("=" * 80 + "\n")
    
    def process_message(self, user_message: str):
        """
        Main pipeline: Process user message through all layers
        """
        print("\n" + "─" * 80)
        print(f"USER: {user_message}")
        print("─" * 80 + "\n")
        
        # LAYER 1: NLU Extraction + LAYER 2: TKG Write
        print("⚙️  LAYER 1 & 2: Extracting entities and updating TKG...")
        extraction_result = self.tkg.ingest_message(user_message, use_llm=self.use_llm)
        
        print(f"   → Extracted {len(extraction_result['extracted']['entities'])} entities")
        print(f"   → Extracted {len(extraction_result['extracted']['relationships'])} relationships")
        
        if extraction_result['extracted']['entities']:
            print("\n   Entities found:")
            for entity in extraction_result['extracted']['entities'][:5]:
                print(f"      • {entity['name']} (type: {entity['type']})")
        
        if extraction_result['extracted']['relationships']:
            print("\n   Relationships found:")
            for rel in extraction_result['extracted']['relationships'][:5]:
                print(f"      • {rel['from']} → {rel['type']} → {rel['to']}")
        
        print()
        
        # LAYER 3: TKG Memory Retrieval
        print("⚙️  LAYER 3: Retrieving relevant facts from TKG...")
        retrieved_facts = self.tkg.query_facts(user_message)
        print(f"   → Retrieved {len(retrieved_facts)} potentially relevant facts\n")
        
        # LAYER 4: Context Engineering + LAYER 5: System Prompt Generation
        print("⚙️  LAYER 4 & 5: Context Engineering & System Prompt Generation...")
        prompt_result = self.context_engineer.build_system_prompt(
            user_message, 
            retrieved_facts
        )
        
        # Display the generated system prompt
        print(format_system_prompt_display(prompt_result))
        
        # LAYER 6: LLM Response (or simulation)
        if self.use_llm and self.api_key:
            print("⚙️  LAYER 6: Generating LLM response (OpenRouter - gpt-oss-20b)...\n")
            response = self._call_llm(prompt_result['system_prompt'], user_message)
        else:
            print("⚙️  LAYER 6: Simulating LLM response (no API key provided)...\n")
            response = self._simulate_response(prompt_result, user_message)
        
        # Display response
        print("─" * 80)
        print("ASSISTANT:")
        print("─" * 80)
        print(response)
        print("─" * 80 + "\n")
        
        # Store in conversation history
        self.conversation_history.append({
            'user': user_message,
            'assistant': response,
            'facts_used': len(prompt_result['facts_used'])
        })
    
    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call OpenRouter API with the engineered system prompt"""
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-20b:free",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"[API Error {response.status_code}: {response.text[:100]}]"
        
        except Exception as e:
            return f"[Error calling LLM: {str(e)}]"
    
    def _simulate_response(self, prompt_result: Dict, user_message: str) -> str:
        """Simulate LLM response based on context (when no API key)"""
        facts = prompt_result['facts_used']
        
        if not facts:
            return "I don't have any relevant information in my memory yet. Could you tell me more?"
        
        # Generate a basic response using the facts
        response_parts = ["Based on what I remember:\n"]
        
        for fact in facts[:3]:
            if fact['type'] == 'relationship':
                response_parts.append(
                    f"- You mentioned that {fact['from']} {fact['relation'].lower().replace('_', ' ')} {fact['to']}"
                )
        
        response_parts.append("\nHow can I help you with this?")
        
        response_parts.append("\n[Note: This is a simulated response. Use --use-llm with --api-key for real LLM responses]")
        
        return "\n".join(response_parts)
    
    def _test_api_key(self):
        """Test if OpenRouter API key is valid"""
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-oss-20b:free",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ OpenRouter API key validated successfully!\n")
            elif response.status_code == 401:
                print("❌ OpenRouter API key is invalid!")
                print("   Get a free key from: https://openrouter.ai/keys\n")
                self.use_llm = False
            else:
                print(f"⚠️  OpenRouter API returned status {response.status_code}")
                print(f"   Continuing anyway...\n")
        
        except requests.exceptions.Timeout:
            print("⚠️  OpenRouter API timeout during validation")
            print("   Network might be slow. Continuing anyway...\n")
        except Exception as e:
            print(f"⚠️  Could not validate API key: {e}")
            print("   Continuing anyway...\n")
    
    def load_demo(self):
        """Load demo conversation"""
        print("\n⚙️  Loading demo conversation...\n")
        demo_messages = create_demo_conversation()
        
        for i, msg in enumerate(demo_messages, 1):
            print(f"[Demo {i}/{len(demo_messages)}] Processing: {msg[:60]}...")
            self.tkg.ingest_message(msg, use_llm=self.use_llm)
        
        print(f"\n✓ Loaded {len(demo_messages)} messages into TKG\n")
        self.show_stats()
    
    def show_stats(self):
        """Show graph statistics"""
        stats = self.tkg.get_graph_stats()
        
        print("\n" + "=" * 80)
        print("📊 TKG STATISTICS")
        print("=" * 80)
        print(f"Total nodes:             {stats['total_nodes']}")
        print(f"Total relationships:     {stats['total_edges']}")
        print(f"Messages processed:      {stats['messages_processed']}")
        print(f"Conversation turns:      {len(self.conversation_history)}")
        
        if stats['node_types']:
            print("\nNode Types:")
            for node_type, count in stats['node_types'].items():
                print(f"  • {node_type:15} {count:3}")
        
        if stats['relationship_types']:
            print("\nRelationship Types:")
            for rel_type, count in stats['relationship_types'].items():
                print(f"  • {rel_type:20} {count:3}")
        
        print("=" * 80 + "\n")
    
    def query_facts(self, query_text: str):
        """Query specific facts"""
        facts = self.tkg.query_facts(query_text)
        
        print("\n" + "=" * 80)
        print(f"🔍 QUERY RESULTS: '{query_text}'")
        print("=" * 80)
        
        if not facts:
            print("No relevant facts found.\n")
            return
        
        print(f"Found {len(facts)} relevant facts:\n")
        
        for i, fact in enumerate(facts[:10], 1):
            if fact['type'] == 'relationship':
                print(f"{i}. {fact['from']} → {fact['relation']} → {fact['to']}")
                print(f"   Time: {fact.get('event_time', 'N/A')[:19]}")
                print(f"   Confidence: {fact.get('confidence', 0.9):.2f}")
            else:
                print(f"{i}. Entity: {fact['entity']} (type: {fact.get('entity_type')})")
                print(f"   Last mentioned: {fact.get('last_mentioned', 'N/A')[:19]}")
            print()
        
        print("=" * 80 + "\n")
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.tkg = TKGMemory(openrouter_api_key=self.api_key)
        self.context_engineer = ContextEngineer(self.tkg)
        self.conversation_history = []
        print("\n✓ Conversation cleared. TKG reset.\n")
    
    def run(self):
        """Main CLI loop"""
        self.print_header()
        
        while True:
            try:
                user_input = input("YOU: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'exit':
                    print("\n👋 Goodbye!\n")
                    break
                
                elif user_input.lower() == 'demo':
                    self.load_demo()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower().startswith('query '):
                    query_text = user_input[6:].strip()
                    self.query_facts(query_text)
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                
                else:
                    # Process as conversation message
                    self.process_message(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Entry point with argument parsing"""
    # The UI prints emoji/unicode throughout. On consoles whose default
    # encoding isn't UTF-8 (e.g. Windows cp1252 when output is piped or
    # redirected) that raises UnicodeEncodeError, so force UTF-8 where possible.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="TKG Memory System - CLI with Context Engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tkg_cli.py                                    # Run without LLM
  python tkg_cli.py --use-llm --api-key YOUR_KEY      # Run with LLM

Get free API key from: https://openrouter.ai/keys
        """
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM for entity extraction and response generation'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenRouter API key (get free key from https://openrouter.ai/keys)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_llm and not args.api_key:
        print("⚠️  Warning: --use-llm requires --api-key")
        print("Get a free API key from: https://openrouter.ai/keys\n")
        response = input("Continue without LLM? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
        args.use_llm = False
    
    # Create and run CLI
    cli = TKGChatCLI(use_llm=args.use_llm, api_key=args.api_key)
    cli.run()


if __name__ == "__main__":
    main()