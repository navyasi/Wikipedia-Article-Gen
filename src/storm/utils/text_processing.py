import re
from typing import List
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Some features may not work.")


class TextProcessor:
    """Utilities for text processing and similarity"""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]]', '', text)
        return text.strip()
    
    def extract_sections_from_text(self, text: str) -> List[str]:
        """Extract sections from Wikipedia-style text"""
        # Simple section extraction based on == headers ==
        sections = re.split(r'\n==+\s*(.+?)\s*==+\n', text)
        return [section.strip() for section in sections if section.strip()]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 3) -> List[tuple]:
        """Find most similar candidates to query"""
        if not candidates:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        candidate_embeddings = self.embedding_model.encode(candidates)
        
        similarities = np.dot(query_embedding, candidate_embeddings.T)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(candidates[i], float(similarities[i])) for i in top_indices]