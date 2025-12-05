# knowledge_loader.py
import os
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re

class JazzKnowledgeBase:
    """Loads and manages jazz theory knowledge for RAG"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
        
        # ChromaDB setup
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=".chromadb"
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("jazz_knowledge")
            print("✅ Loaded existing knowledge base")
        except:
            self.collection = self.client.create_collection("jazz_knowledge")
            self._load_all_knowledge()
            print("✅ Created new knowledge base")
    
    def _load_all_knowledge(self):
        """Load all markdown files into vector database"""
        print("📚 Loading knowledge files...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Walk through knowledge directory
        for md_file in self.knowledge_dir.rglob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            
            # Split into sections (by headers)
            sections = self._split_into_sections(content)
            
            for i, section in enumerate(sections):
                doc_id = f"{md_file.stem}_{i}"
                documents.append(section['content'])
                metadatas.append({
                    'source': str(md_file),
                    'title': section['title'],
                    'category': md_file.parent.name
                })
                ids.append(doc_id)
        
        # Add to ChromaDB
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Loaded {len(documents)} knowledge sections")
    
    def _split_into_sections(self, content: str) -> List[Dict]:
        """Split markdown content by headers"""
        sections = []
        lines = content.split('\n')
        
        current_title = "Introduction"
        current_content = []
        
        for line in lines:
            # Check if it's a header
            if line.startswith('##'):
                # Save previous section
                if current_content:
                    sections.append({
                        'title': current_title,
                        'content': '\n'.join(current_content).strip()
                    })
                # Start new section
                current_title = line.replace('#', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            sections.append({
                'title': current_title,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search knowledge base for relevant information"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        relevant_docs = []
        for i in range(len(results['documents'][0])):
            relevant_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'relevance': 1 - results['distances'][0][i]  # Convert distance to relevance
            })
        
        return relevant_docs
    
    def get_context_for_analysis(self, audio_features: Dict, jazz_analysis: Dict) -> str:
        """Get relevant knowledge based on audio analysis"""
        # Build search queries
        queries = []
        
        # Tempo-based query
        tempo_cat = jazz_analysis['tempo_category']
        if 'Bebop' in tempo_cat or 'Fast' in tempo_cat:
            queries.append("bebop techniques fast tempo improvisation")
        elif 'Ballad' in tempo_cat:
            queries.append("ballad playing melodic development phrasing")
        else:
            queries.append("swing medium tempo improvisation")
        
        # Rhythm-based query
        if audio_features['rhythm_complexity'] > 6:
            queries.append("complex rhythms syncopation bebop")
        
        # Always get scale info
        queries.append("scales chord progressions jazz theory")
        
        # Search knowledge base
        all_context = []
        for query in queries:
            results = self.search(query, n_results=2)
            for result in results:
                all_context.append(f"## {result['metadata']['title']}\n{result['content'][:500]}...")
        
        return "\n\n".join(all_context[:4])  # Max 4 sections

# Singleton instance
_kb_instance = None

def get_knowledge_base():
    """Get or create knowledge base singleton"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = JazzKnowledgeBase()
    return _kb_instance
