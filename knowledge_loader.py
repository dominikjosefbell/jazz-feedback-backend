"""
Jazz Knowledge Base Loader - RAG System
Uses ChromaDB + Sentence Transformers for semantic search
"""

import os
import glob
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class JazzKnowledgeBase:
    """
    Retrieval Augmented Generation system for jazz theory knowledge.
    Loads markdown files from knowledge/ directory and enables semantic search.
    """
    
    def __init__(self, knowledge_dir: str = "knowledge", persist_dir: str = ".chromadb"):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_dir: Directory containing knowledge markdown files
            persist_dir: Directory to persist ChromaDB database
        """
        self.knowledge_dir = knowledge_dir
        self.persist_dir = persist_dir
        
        # Initialize embedding model (lightweight, good quality)
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        
        # Initialize ChromaDB
        print("🔄 Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="jazz_knowledge",
            metadata={"description": "Jazz theory and improvisation knowledge base"}
        )
        
        print(f"✅ ChromaDB initialized ({self.collection.count()} documents)")
        
        # Load knowledge if collection is empty
        if self.collection.count() == 0:
            print("📚 Knowledge base empty - loading files...")
            self.load_knowledge()
        else:
            print("✅ Using existing knowledge base")
    
    def load_knowledge(self):
        """Load all markdown files from knowledge directory into ChromaDB."""
        if not os.path.exists(self.knowledge_dir):
            print(f"⚠️ Knowledge directory not found: {self.knowledge_dir}")
            print("   Creating empty knowledge base...")
            return
        
        # Find all markdown files
        md_files = glob.glob(f"{self.knowledge_dir}/**/*.md", recursive=True)
        
        if not md_files:
            print(f"⚠️ No markdown files found in {self.knowledge_dir}")
            return
        
        print(f"📖 Found {len(md_files)} markdown files")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for file_path in md_files:
            print(f"   Loading: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by headers (## sections)
            sections = self._split_by_headers(content, file_path)
            
            for section_id, section in enumerate(sections):
                doc_id = f"{file_path}_{section_id}"
                
                all_documents.append(section['content'])
                all_metadatas.append({
                    'source': file_path,
                    'title': section['title'],
                    'section_id': section_id
                })
                all_ids.append(doc_id)
        
        # Add to ChromaDB in batch
        if all_documents:
            print(f"💾 Adding {len(all_documents)} sections to ChromaDB...")
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"✅ Knowledge base loaded! ({len(all_documents)} sections)")
        else:
            print("⚠️ No content to add")
    
    def _split_by_headers(self, content: str, file_path: str) -> List[Dict]:
        """
        Split markdown content by ## headers into sections.
        
        Args:
            content: Markdown file content
            file_path: Path to source file
            
        Returns:
            List of dictionaries with 'title' and 'content'
        """
        sections = []
        current_section = {'title': 'Introduction', 'content': ''}
        
        lines = content.split('\n')
        
        for line in lines:
            # Check for ## header (not #)
            if line.startswith('## ') and not line.startswith('### '):
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': line.replace('##', '').strip(),
                    'content': ''
                }
            else:
                # Add line to current section
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        # If no sections found, treat whole file as one section
        if not sections:
            filename = os.path.basename(file_path).replace('.md', '').replace('_', ' ').title()
            sections.append({
                'title': filename,
                'content': content
            })
        
        return sections
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Semantic search in knowledge base.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if self.collection.count() == 0:
            print("⚠️ Knowledge base is empty")
            return []
        
        # Search
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        # Format results
        formatted_results = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'source': results['metadatas'][0][i].get('source', 'unknown'),
                    'title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'distance': results['distances'][0][i] if 'distances' in results else 0.0
                })
        
        return formatted_results
    
    def get_context_for_analysis(self, 
                                  tempo: float, 
                                  tempo_category: str,
                                  rhythm_complexity: float) -> str:
        """
        Get relevant jazz theory context based on audio analysis.
        
        Args:
            tempo: BPM tempo
            tempo_category: Category like "Bebop", "Ballad", etc.
            rhythm_complexity: Complexity score 0-10
            
        Returns:
            Formatted context string for AI prompt
        """
        context_parts = []
        
        # Search based on tempo category
        if "bebop" in tempo_category.lower() or "fast" in tempo_category.lower():
            results = self.search("bebop techniques chromatic approach fast tempo", n_results=2)
            context_parts.extend(results)
        elif "ballad" in tempo_category.lower() or "slow" in tempo_category.lower():
            results = self.search("ballad phrasing melodic development space", n_results=2)
            context_parts.extend(results)
        elif "modal" in tempo_category.lower():
            results = self.search("modal scales dorian improvisation", n_results=2)
            context_parts.extend(results)
        
        # Search based on rhythm complexity
        if rhythm_complexity > 7:
            results = self.search("syncopation rhythmic displacement polyrhythm", n_results=1)
            context_parts.extend(results)
        elif rhythm_complexity < 4:
            results = self.search("rhythm practice simple patterns", n_results=1)
            context_parts.extend(results)
        
        # Always add some scale/chord info
        results = self.search("ii-V-I progression scales", n_results=1)
        context_parts.extend(results)
        
        # Format context
        if not context_parts:
            return ""
        
        # Remove duplicates
        seen = set()
        unique_parts = []
        for part in context_parts:
            key = part['source'] + part['title']
            if key not in seen:
                seen.add(key)
                unique_parts.append(part)
        
        # Format as text
        formatted = "\n\n=== RELEVANT JAZZ THEORY CONTEXT ===\n\n"
        for part in unique_parts[:3]:  # Max 3 sections to keep prompt reasonable
            formatted += f"## {part['title']}\n"
            formatted += f"(Source: {os.path.basename(part['source'])})\n\n"
            # Limit content length
            content = part['content'][:500] + "..." if len(part['content']) > 500 else part['content']
            formatted += content + "\n\n"
        
        formatted += "=== END CONTEXT ===\n"
        
        return formatted


# Global instance (lazy loaded)
_knowledge_base = None

def get_knowledge_base() -> JazzKnowledgeBase:
    """
    Get or create global knowledge base instance.
    Lazy loading to avoid initialization on import.
    """
    global _knowledge_base
    if _knowledge_base is None:
        print("🎷 Initializing Jazz Knowledge Base...")
        _knowledge_base = JazzKnowledgeBase()
    return _knowledge_base


# Example usage
if __name__ == "__main__":
    # Test the knowledge base
    kb = get_knowledge_base()
    
    print("\n=== Testing Search ===\n")
    
    # Test searches
    queries = [
        "bebop scales over dominant chords",
        "ii-V-I progression techniques",
        "how to practice improvisation"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = kb.search(query, n_results=2)
        for i, result in enumerate(results):
            print(f"\n  Result {i+1}:")
            print(f"  Title: {result['title']}")
            print(f"  Source: {result['source']}")
            print(f"  Content preview: {result['content'][:150]}...")
    
    print("\n=== Testing Context Generation ===\n")
    
    # Test context for fast bebop
    context = kb.get_context_for_analysis(
        tempo=200.0,
        tempo_category="Fast Bebop",
        rhythm_complexity=8.5
    )
    print("Context for fast bebop tempo:")
    print(context[:500] + "...")
