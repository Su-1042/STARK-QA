import os
import logging
import time
import random
from typing import Dict, List, Any

# Import config for retry parameters
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    import config
    RETRY_MAX_ATTEMPTS = getattr(config, 'API_RETRY_MAX_ATTEMPTS', 3)
    RETRY_INITIAL_DELAY = getattr(config, 'API_RETRY_INITIAL_DELAY', 1.0)
    RETRY_EXPONENTIAL_BASE = getattr(config, 'API_RETRY_EXPONENTIAL_BASE', 2)
except ImportError:
    # Fallback to default values if config import fails
    RETRY_MAX_ATTEMPTS = 3
    RETRY_INITIAL_DELAY = 1.0
    RETRY_EXPONENTIAL_BASE = 2

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    try:
        from mistralai.client import MistralClient
        MISTRAL_AVAILABLE = True
        LEGACY_MISTRAL = True
    except ImportError:
        MISTRAL_AVAILABLE = False
        print("Warning: Mistral AI client not available. Install with: pip install mistralai")

class StarkQASystem:
    """STARK-QA: Combined RAG-KAG system using Mistral API."""
    
    def __init__(self, rag_system, kg_builder):
        self.rag_system = rag_system
        self.kg_builder = kg_builder
        self.logger = logging.getLogger(__name__)
        
        # Initialize Mistral client
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            self.logger.warning("MISTRAL_API_KEY not found in environment variables")
            self.client = None
            return
        
        if not MISTRAL_AVAILABLE:
            self.logger.warning("Mistral client not available")
            self.client = None
            return
        
        try:
            # Try new Mistral client first
            self.client = Mistral(api_key=api_key)
            self.model = "mistral-small"  # Use available model
            self.use_legacy = False
        except (NameError, Exception):
            try:
                # Fallback to legacy client
                self.client = MistralClient(api_key=api_key)
                self.model = "mistral-small"
                self.use_legacy = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Mistral client: {e}")
                self.client = None
        
    def retrieve_combined_context(self, question: str, rag_weight: float = 0.7) -> Dict[str, str]:
        """Retrieve context from both RAG and KAG systems with intelligent fusion."""
        # Get RAG context (more chunks for better coverage)
        rag_context = self.rag_system.retrieve_context(question, top_k=10)
        
        # Get KAG context
        kg_context = self.kg_builder.retrieve_context(question)
        
        # Intelligent context fusion based on question type
        combined_context = self._smart_combine_contexts(question, rag_context, kg_context, rag_weight)
        
        return {
            'rag_context': rag_context,
            'kg_context': kg_context,
            'combined_context': combined_context
        }
    
    def _smart_combine_contexts(self, question: str, rag_context: str, kg_context: str, rag_weight: float = 0.7) -> str:
        """Intelligently combine RAG and KAG contexts based on question type."""
        question_lower = question.lower()
        context_parts = []
        
        # For "who" questions, prioritize KG (entity information)
        if question_lower.startswith('who'):
            if kg_context and kg_context.strip():
                context_parts.append(f"Character Info: {kg_context}")
            if rag_context and rag_context.strip():
                context_parts.append(f"Story Context: {rag_context}")
        
        # For "what" questions, balance both equally
        elif question_lower.startswith('what'):
            if rag_context and rag_context.strip():
                context_parts.append(f"Story Events: {rag_context}")
            if kg_context and kg_context.strip():
                context_parts.append(f"Entity Details: {kg_context}")
        
        # For "where" questions, prioritize RAG (descriptive text)
        elif question_lower.startswith('where'):
            if rag_context and rag_context.strip():
                context_parts.append(f"Location Context: {rag_context}")
            if kg_context and kg_context.strip():
                context_parts.append(f"Additional Info: {kg_context}")
        
        # For "why" questions, prioritize RAG (causal relationships)
        elif question_lower.startswith('why'):
            if rag_context and rag_context.strip():
                context_parts.append(f"Explanation: {rag_context}")
            if kg_context and kg_context.strip():
                context_parts.append(f"Background: {kg_context}")
        
        # Default: combine with RAG priority
        else:
            if rag_context and rag_context.strip():
                context_parts.append(f"Primary Context: {rag_context}")
            if kg_context and kg_context.strip():
                context_parts.append(f"Supporting Info: {kg_context}")
        
        return "\n\n".join(context_parts) if context_parts else "Limited context available."
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Mistral API with comprehensive fallback."""
        # If no API client, use rule-based fallback
        if not self.client:
            return self._generate_fallback_answer(question, context)
        
        try:
            # Construct prompt
            prompt = self._construct_prompt(question, context)
            
            # Try multiple API call formats
            try:
                # Format 1: New Mistral client with chat.complete
                if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'complete'):
                    response = self._api_call_with_retry(lambda: self.client.chat.complete(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,  # More deterministic
                        max_tokens=25  # Very short responses
                    ))
                    return response.choices[0].message.content.strip()
                
                # Format 2: Legacy chat method  
                elif hasattr(self.client, 'chat'):
                    response = self._api_call_with_retry(lambda: self.client.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=25
                    ))
                    return response.choices[0].message.content.strip()
                
                # Format 3: Direct complete method
                elif hasattr(self.client, 'complete'):
                    response = self._api_call_with_retry(lambda: self.client.complete(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=25
                    ))
                    return response.choices[0].text.strip()
                
            except Exception as api_error:
                self.logger.error(f"API call failed: {api_error}")
                return self._generate_fallback_answer(question, context)
                
        except Exception as e:
            self.logger.error(f"Error in generate_answer: {e}")
            return self._generate_fallback_answer(question, context)
    
    def _generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate rule-based answer when API fails."""
        if not context or context.strip() == "":
            return "I don't have enough context to answer this question."
        
        question_lower = question.lower()
        
        # Extract key information from context
        context_lower = context.lower()
        
        # Question-type specific responses with better context analysis
        if question_lower.startswith('who'):
            # Look for names/characters in context
            import re
            names = re.findall(r'\b[A-Z][a-z]+\b', context)
            if names:
                unique_names = list(dict.fromkeys(names))  # Remove duplicates, preserve order
                return unique_names[0]  # Just return the first name
            
            # Look for character descriptions
            character_words = ['man', 'woman', 'king', 'queen', 'cook', 'tailor', 'weaver']
            for word in character_words:
                if word in context.lower():
                    return f"the {word}"
            
            return "the character"
            
        elif question_lower.startswith('what'):
            # Look for objects, actions, or events - extract key content words
            words = context.split()
            content_words = []
            
            question_words = set(question_lower.split()) - {'what', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'did', 'do', 'does'}
            
            # Find words related to the question
            for word in words:
                if (word.lower() in question_words or 
                    len(word) > 3 and 
                    word.lower() not in ['this', 'that', 'they', 'were', 'with', 'from', 'have']):
                    content_words.append(word)
            
            if content_words:
                # Return first few content words
                return ' '.join(content_words[:3])
            
            return "something mentioned"
            
        elif question_lower.startswith('where'):
            # Look for location indicators
            location_words = ['cottage', 'castle', 'forest', 'kingdom', 'village', 'house', 'palace', 'mountain', 'river', 'cave', 'tower', 'farmhouse', 'kitchen', 'room']
            for location in location_words:
                if location in context.lower():
                    # Find the phrase containing this location
                    context_lower = context.lower()
                    if f"in the {location}" in context_lower:
                        return f"in the {location}"
                    elif f"a {location}" in context_lower:
                        return f"a {location}"
                    elif f"the {location}" in context_lower:
                        return f"the {location}"
                    else:
                        return location
            
            # Look for prepositions indicating location
            location_patterns = [r'in the \w+', r'at the \w+', r'by the \w+']
            for pattern in location_patterns:
                matches = re.findall(pattern, context.lower())
                if matches:
                    return matches[0]
            
            return "somewhere mentioned"
            
        elif question_lower.startswith('why'):
            # Look for causal indicators
            causal_words = ['because', 'since', 'due to', 'as a result', 'therefore', 'so that']
            context_sentences = context.split('.')
            
            for sentence in context_sentences:
                for causal in causal_words:
                    if causal in sentence.lower():
                        # Extract the reason part
                        parts = sentence.lower().split(causal, 1)
                        if len(parts) > 1:
                            reason = parts[1].strip()
                            words = reason.split()[:5]  # Take first 5 words
                            return ' '.join(words)
            
            # Look for descriptive adjectives or states
            descriptive_words = ['old', 'young', 'scared', 'hungry', 'tired', 'beautiful', 'pleased', 'disappointed']
            for word in descriptive_words:
                if word in context.lower():
                    return word
            
            return "reason mentioned"
            
        elif question_lower.startswith('how'):
            # Check if it's asking about feelings
            if 'feel' in question_lower or 'felt' in question_lower:
                emotion_words = ['happy', 'sad', 'angry', 'scared', 'frightened', 'pleased', 'disappointed', 'terrified', 'delighted']
                for word in emotion_words:
                    if word in context.lower():
                        return word
                return "emotional"
            
            # Look for process or method descriptions
            method_words = ['by', 'with', 'using', 'through', 'via']
            context_sentences = context.split('.')
            
            for sentence in context_sentences:
                for method in method_words:
                    if method in sentence.lower():
                        # Extract method part
                        parts = sentence.lower().split(method, 1)
                        if len(parts) > 1:
                            method_part = parts[1].strip()
                            words = method_part.split()[:4]  # Take first 4 words
                            return f"{method} {' '.join(words)}"
            
            return "method described"
            
        else:
            # General answer - find most relevant content words
            question_words = set(question_lower.split()) - {'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but'}
            
            # Extract content words from context that match question
            context_words = context.split()
            relevant_words = []
            
            for word in context_words:
                if (word.lower() in question_words or 
                    (len(word) > 3 and word.lower() not in ['this', 'that', 'they', 'were', 'with', 'from'])):
                    relevant_words.append(word)
            
            if relevant_words:
                return ' '.join(relevant_words[:4])  # Take first 4 relevant words
            
            # Last resort: find first meaningful content
            sentences = context.split('.')
            for sentence in sentences:
                words = sentence.split()
                if len(words) > 3:
                    content_words = [w for w in words[:6] if len(w) > 2]
                    if content_words:
                        return ' '.join(content_words[:3])
            
            return "information provided"
    
    def _construct_prompt(self, question: str, context: str) -> str:
        """Construct prompt for Mistral API."""
        prompt = f"""Based on the story context, provide a very short, direct answer to the question.

Context: {context}

Question: {question}

Requirements:
- Answer in 1-6 words maximum
- Be direct and factual
- No explanations or extra context
- Extract the exact information asked for
- Use the same style as the original story text

Answer:"""
        
        return prompt
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer a question using STARK-QA."""
        # Retrieve combined context
        contexts = self.retrieve_combined_context(question)
        
        # Generate answer
        raw_answer = self.generate_answer(question, contexts['combined_context'])
        processed_answer = raw_answer.strip()
        if processed_answer.endswith('.'):
            pass
        else:
            processed_answer += '.'
        
        return {
            'question': question,
            'answer': processed_answer,
            'raw_answer': raw_answer,  # Keep for debugging
            'rag_context': contexts['rag_context'],
            'kg_context': contexts['kg_context'],
            'combined_context': contexts['combined_context']
        }

    def _api_call_with_retry(self, api_call_func, max_retries: int = None, initial_delay: float = None):
        """Execute API call with retry logic for rate limiting (429 errors)."""
        # Use configured values or defaults
        if max_retries is None:
            max_retries = RETRY_MAX_ATTEMPTS
        if initial_delay is None:
            initial_delay = RETRY_INITIAL_DELAY
            
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff and jitter
                        delay = initial_delay * (RETRY_EXPONENTIAL_BASE ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Rate limit exceeded after {max_retries + 1} attempts")
                        raise e
                else:
                    # For non-rate-limit errors, raise immediately
                    raise e
        
        # If we get here, all retries failed
        raise last_exception