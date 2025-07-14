import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import optional dependencies with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available")

class RobustRAG:
    """Robust RAG system that works even with limited dependencies."""
    
    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize models with fallbacks
        self._init_models()
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced for stability
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Storage
        self.stories = {}
        self.story_chunks = []
        self.chunk_metadata = []
        
        # Indices
        self.dense_index = None
        self.dense_embeddings_matrix = None
        self.tfidf_matrix = None
        
    def _init_models(self):
        """Initialize embedding models with fallbacks."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.logger.info("Initializing sentence transformers...")
                # Use smaller, more compatible models
                self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embeddings_available = True
            else:
                self.logger.warning("Sentence transformers not available, using TF-IDF only")
                self.dense_model = None
                self.embeddings_available = False
        except Exception as e:
            self.logger.error(f"Error initializing embedding models: {e}")
            self.logger.info("Falling back to TF-IDF only mode")
            self.dense_model = None
            self.embeddings_available = False
    
    def _create_simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create simple embeddings using TF-IDF as fallback."""
        self.logger.info("Creating simple TF-IDF embeddings...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix.toarray()
    
    def chunk_story(self, story_text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
        """Split story into overlapping chunks with smaller size for stability."""
        words = story_text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
                
        return chunks
    
    def index_stories(self, stories: Dict[str, str]):
        """Index stories with robust error handling."""
        self.logger.info("Indexing stories with robust RAG...")
        
        cache_file = os.path.join(self.cache_dir, "robust_rag_index.json")
        
        if os.path.exists(cache_file):
            self.logger.info("Loading cached RAG index...")
            try:
                self._load_cache(cache_file)
                return
            except Exception as e:
                self.logger.warning(f"Cache loading failed: {e}, rebuilding index...")
        
        self.stories = stories
        
        # Create chunks
        for story_name, story_text in stories.items():
            chunks = self.chunk_story(story_text)
            for i, chunk in enumerate(chunks):
                self.story_chunks.append(chunk)
                self.chunk_metadata.append({
                    'story_name': story_name,
                    'chunk_id': i,
                    'chunk_text': chunk
                })
        
        self.logger.info(f"Created {len(self.story_chunks)} chunks from {len(stories)} stories")
        
        # Build embeddings with error handling
        if self.embeddings_available:
            try:
                self.logger.info("Creating dense embeddings...")
                dense_embeddings = self.dense_model.encode(
                    self.story_chunks, 
                    batch_size=4,  # Very small batch size for stability
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                
                # Normalize embeddings
                dense_embeddings = dense_embeddings / np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
                
                # Try to use FAISS if available
                if FAISS_AVAILABLE:
                    try:
                        dimension = dense_embeddings.shape[1]
                        self.dense_index = faiss.IndexFlatIP(dimension)
                        self.dense_index.add(dense_embeddings.astype('float32'))
                        self.logger.info("FAISS index created successfully")
                    except Exception as e:
                        self.logger.warning(f"FAISS indexing failed: {e}, using numpy fallback")
                        self.dense_embeddings_matrix = dense_embeddings
                else:
                    self.dense_embeddings_matrix = dense_embeddings
                    
            except Exception as e:
                self.logger.error(f"Dense embedding creation failed: {e}")
                self.logger.info("Falling back to TF-IDF only")
                self.embeddings_available = False
        
        # Build TF-IDF index (always works)
        self.logger.info("Building TF-IDF index...")
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.story_chunks)
            self.logger.info("TF-IDF index created successfully")
        except Exception as e:
            self.logger.error(f"TF-IDF indexing failed: {e}")
            raise e
        
        # Cache the index
        try:
            self._save_cache(cache_file)
        except Exception as e:
            self.logger.warning(f"Cache saving failed: {e}")
        
    def _save_cache(self, cache_file: str):
        """Save index to cache with error handling."""
        cache_data = {
            'stories': self.stories,
            'chunk_metadata': self.chunk_metadata,
            'story_chunks': self.story_chunks,
            'embeddings_available': self.embeddings_available
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Save embeddings if available
        if self.dense_embeddings_matrix is not None:
            np.save(cache_file.replace('.json', '_embeddings.npy'), self.dense_embeddings_matrix)
            
        self.logger.info("Saved RAG index to cache")
    
    def _load_cache(self, cache_file: str):
        """Load index from cache with error handling."""
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        self.stories = cache_data['stories']
        self.chunk_metadata = cache_data['chunk_metadata']
        self.story_chunks = cache_data['story_chunks']
        self.embeddings_available = cache_data.get('embeddings_available', False)
        
        # Load embeddings if available
        embeddings_file = cache_file.replace('.json', '_embeddings.npy')
        if os.path.exists(embeddings_file):
            try:
                self.dense_embeddings_matrix = np.load(embeddings_file)
                self.embeddings_available = True
            except Exception as e:
                self.logger.warning(f"Failed to load embeddings: {e}")
                self.embeddings_available = False
        
        # Rebuild TF-IDF (quick operation)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.story_chunks)
        
    def dense_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Dense retrieval with fallbacks."""
        if not self.embeddings_available or self.dense_model is None:
            return []
        
        try:
            query_embedding = self.dense_model.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            if self.dense_index is not None:
                # Use FAISS
                scores, indices = self.dense_index.search(query_embedding.astype('float32'), top_k)
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.story_chunks):
                        results.append((self.story_chunks[idx], float(score)))
            elif self.dense_embeddings_matrix is not None:
                # Use numpy
                similarities = np.dot(query_embedding, self.dense_embeddings_matrix.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k]
                results = [(self.story_chunks[idx], float(similarities[idx])) for idx in top_indices]
            else:
                results = []
                
            return results
            
        except Exception as e:
            self.logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def sparse_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Sparse retrieval using TF-IDF."""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append((self.story_chunks[idx], float(scores[idx])))
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def retrieve_context(self, question: str, top_k: int = 5) -> str:
        """Retrieve relevant context with multiple fallback strategies and question-specific filtering."""
        all_results = []
        
        # Try dense retrieval first
        if self.embeddings_available:
            dense_results = self.dense_retrieval(question, top_k * 2)  # Get more candidates
            all_results.extend(dense_results)
        
        # Always try sparse retrieval
        sparse_results = self.sparse_retrieval(question, top_k * 2)
        all_results.extend(sparse_results)
        
        # Remove duplicates and get unique chunks
        seen_chunks = set()
        unique_results = []
        for chunk, score in all_results:
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                unique_results.append((chunk, score))
        
        # Filter results based on question type for better relevance
        filtered_results = self._filter_by_question_type(question, unique_results)
        
        # Sort by score and take top results
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in filtered_results[:top_k]]
        
        return "\n\n".join(top_chunks) if top_chunks else "No relevant context found."
    
    def _filter_by_question_type(self, question: str, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Filter and re-rank results based on question type."""
        q_lower = question.lower()
        filtered = []
        
        # Extract key terms from question
        question_terms = set(q_lower.split()) - {'who', 'what', 'where', 'when', 'why', 'how', 'did', 'do', 'does', 'is', 'are', 'was', 'were', 'the', 'a', 'an'}
        
        for chunk, score in results:
            chunk_lower = chunk.lower()
            
            # Boost score if chunk contains question terms
            term_matches = sum(1 for term in question_terms if term in chunk_lower)
            boosted_score = score + (term_matches * 0.1)
            
            # Question-type specific filtering
            if q_lower.startswith('who'):
                # Prioritize chunks with character names or descriptions
                if any(marker in chunk_lower for marker in ['king', 'queen', 'man', 'woman', 'cook', 'tailor', 'weaver', 'smith', 'miller']):
                    boosted_score += 0.2
                # Look for proper names (capitalized words)
                if any(word[0].isupper() for word in chunk.split() if len(word) > 2):
                    boosted_score += 0.15
            
            elif q_lower.startswith('what'):
                # Prioritize chunks with objects, actions, or events
                if any(marker in chunk_lower for marker in ['animals', 'cows', 'hens', 'bannock', 'cottage', 'doing', 'made', 'took']):
                    boosted_score += 0.2
            
            elif q_lower.startswith('where'):
                # Prioritize chunks with location indicators
                if any(marker in chunk_lower for marker in ['cottage', 'house', 'castle', 'forest', 'village', 'in', 'at', 'by', 'near']):
                    boosted_score += 0.2
            
            elif q_lower.startswith('why'):
                # Prioritize chunks with causal language
                if any(marker in chunk_lower for marker in ['because', 'since', 'so', 'wanted', 'needed', 'decided']):
                    boosted_score += 0.2
            
            elif q_lower.startswith('how') and ('feel' in q_lower or 'felt' in q_lower):
                # Prioritize chunks with emotion words
                if any(marker in chunk_lower for marker in ['scared', 'frightened', 'pleased', 'disappointed', 'terrified', 'delighted', 'happy', 'sad']):
                    boosted_score += 0.3
            
            # Only include chunks with reasonable relevance
            if boosted_score > 0.05:  # Threshold for relevance
                filtered.append((chunk, boosted_score))
        
        return filtered
    
    def answer_question(self, question: str) -> str:
        """Basic answer generation for baseline comparison."""
        context = self.retrieve_context(question)
        return f"Based on the story context, this question relates to: {context[:200]}..."