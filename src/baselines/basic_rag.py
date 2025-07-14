import os
import logging
import re
from typing import Dict, List, Any
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # These will be installed via conda environment
    pass

class BasicRAG:
    """Basic RAG baseline using simple TF-IDF retrieval."""
    
    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize simple models
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        
        # Storage
        self.stories = {}
        self.story_chunks = []
        self.tfidf_matrix = None
        
    def chunk_story(self, story_text: str, chunk_size: int = 300) -> List[str]:
        """Simple chunking without overlap."""
        words = story_text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
                
        return chunks
    
    def index_stories(self, stories: Dict[str, str]):
        """Index stories using basic TF-IDF."""
        self.logger.info("Indexing stories with basic RAG...")
        
        self.stories = stories
        
        # Create chunks
        for story_name, story_text in stories.items():
            chunks = self.chunk_story(story_text)
            self.story_chunks.extend(chunks)
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.story_chunks)
        
        self.logger.info(f"Indexed {len(self.story_chunks)} chunks")
    
    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        """Retrieve context using TF-IDF similarity."""
        if self.tfidf_matrix is None:
            return ""
        
        # Transform question
        question_vector = self.vectorizer.transform([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()
        
        # Get top-k chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0:
                relevant_chunks.append(self.story_chunks[idx])
        
        return "\n\n".join(relevant_chunks)
    
    def answer_question(self, question: str) -> str:
        """Generate a basic answer."""
        context = self.retrieve_context(question)
        
        # Simple extractive answer (find sentence with highest overlap)
        if not context:
            return "I don't have enough information to answer this question."
        
        sentences = context.split('.')
        question_words = set(question.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence.strip()
        
        return best_sentence if best_sentence else "Unable to find a relevant answer."