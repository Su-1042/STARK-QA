import os
import logging
from typing import Dict, List, Any

class SimpleFallbackQA:
    """Simple fallback QA system that works without API."""
    
    def __init__(self, rag_system, kg_builder):
        self.rag_system = rag_system
        self.kg_builder = kg_builder
        self.logger = logging.getLogger(__name__)
        
    def retrieve_combined_context(self, question: str, rag_weight: float = 0.6) -> Dict[str, str]:
        """Retrieve context from both RAG and KAG systems."""
        # Get RAG context
        rag_context = self.rag_system.retrieve_context(question, top_k=5)
        
        # Get KAG context
        kg_context = self.kg_builder.retrieve_context(question)
        
        return {
            'rag_context': rag_context,
            'kg_context': kg_context,
            'combined_context': self._combine_contexts(rag_context, kg_context, rag_weight)
        }
    
    def _combine_contexts(self, rag_context: str, kg_context: str, rag_weight: float = 0.6) -> str:
        """Intelligently combine RAG and KAG contexts."""
        context_parts = []
        
        if rag_context.strip():
            context_parts.append(f"Story Context:\n{rag_context}")
        
        if kg_context.strip():
            context_parts.append(f"Knowledge Graph Context:\n{kg_context}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate simple rule-based answer."""
        if not context.strip():
            return "I don't have enough context to answer this question."
        
        # Simple rule-based answer generation
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Extract potential answers based on question type
        if question_lower.startswith('who'):
            return self._extract_who_answer(context)
        elif question_lower.startswith('what'):
            return self._extract_what_answer(context, question_lower)
        elif question_lower.startswith('where'):
            return self._extract_where_answer(context)
        elif question_lower.startswith('when'):
            return self._extract_when_answer(context)
        elif question_lower.startswith('why'):
            return self._extract_why_answer(context)
        elif question_lower.startswith('how'):
            return self._extract_how_answer(context)
        else:
            # General answer - return first meaningful sentence
            sentences = context.split('. ')
            for sentence in sentences[:3]:
                if len(sentence.strip()) > 10:
                    return sentence.strip() + '.'
            return "Based on the context, this relates to the story events described."
    
    def _extract_who_answer(self, context: str) -> str:
        """Extract 'who' answer from context."""
        # Look for character names (capitalized words)
        import re
        characters = re.findall(r'\b[A-Z][a-z]+\b', context)
        if characters:
            return f"This involves {characters[0]}."
        return "The answer involves characters from the story."
    
    def _extract_what_answer(self, context: str, question: str) -> str:
        """Extract 'what' answer from context."""
        sentences = context.split('. ')
        for sentence in sentences[:2]:
            if len(sentence.strip()) > 15:
                return sentence.strip() + '.'
        return "This relates to events described in the story."
    
    def _extract_where_answer(self, context: str) -> str:
        """Extract 'where' answer from context."""
        # Look for location words
        locations = ['castle', 'forest', 'kingdom', 'village', 'house', 'palace']
        context_lower = context.lower()
        for location in locations:
            if location in context_lower:
                return f"This takes place in or near a {location}."
        return "The location is described in the story context."
    
    def _extract_when_answer(self, context: str) -> str:
        """Extract 'when' answer from context."""
        time_words = ['once upon a time', 'long ago', 'yesterday', 'today', 'tomorrow']
        context_lower = context.lower()
        for time_word in time_words:
            if time_word in context_lower:
                return f"This happened {time_word}."
        return "The timing is described in the story."
    
    def _extract_why_answer(self, context: str) -> str:
        """Extract 'why' answer from context."""
        # Look for causal words
        causal_words = ['because', 'since', 'due to', 'as a result']
        sentences = context.split('. ')
        for sentence in sentences:
            for causal in causal_words:
                if causal in sentence.lower():
                    return sentence.strip() + '.'
        return "The reason is explained in the story context."
    
    def _extract_how_answer(self, context: str) -> str:
        """Extract 'how' answer from context."""
        # Look for method/process descriptions
        sentences = context.split('. ')
        for sentence in sentences[:2]:
            if any(word in sentence.lower() for word in ['by', 'with', 'using', 'through']):
                return sentence.strip() + '.'
        return "The method is described in the story."
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer a question using simple fallback."""
        # Retrieve combined context
        contexts = self.retrieve_combined_context(question)
        
        # Generate answer
        answer = self.generate_answer(question, contexts['combined_context'])
        
        return {
            'question': question,
            'answer': answer,
            'rag_context': contexts['rag_context'],
            'kg_context': contexts['kg_context'],
            'combined_context': contexts['combined_context']
        }