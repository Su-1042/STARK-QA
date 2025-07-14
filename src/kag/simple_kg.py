import os
import json
import re
import logging
from typing import Dict, List, Any, Set
from collections import defaultdict

class SimpleKnowledgeGraph:
    """Simple knowledge graph that actually works and provides context."""
    
    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Simple storage
        self.entities = {}  # entity -> {type, stories, mentions}
        self.relationships = []  # list of (subject, predicate, object, story)
        self.stories = {}
        
        # For context retrieval
        self.entity_contexts = defaultdict(list)
        
    def extract_simple_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using comprehensive patterns."""
        entities = []
        text_lower = text.lower()
        
        # Expanded character patterns - find proper nouns and character references
        character_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Multi-word names
            r'\b(?:king|queen|prince|princess|lord|lady|sir|master|goodman|goodwife|old\s+(?:man|woman)|young\s+(?:man|woman))\b',
            r'\b(?:tailor|smith|miller|weaver|shepherd|cook|huntsman|guard|soldier|knight)\b',
            r'\b(?:father|mother|son|daughter|brother|sister|wife|husband|child|children)\b'
        ]
        
        # Find all character mentions
        character_entities = set()
        for pattern in character_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0).strip()
                # Filter out common words that aren't names
                if (entity_text.lower() not in ['the', 'and', 'but', 'for', 'with', 'once', 'upon', 'time', 
                                               'there', 'was', 'were', 'had', 'has', 'this', 'that', 'they', 'them'] 
                    and len(entity_text) > 2):
                    character_entities.add(entity_text.lower())
        
        # Add character entities
        for entity in character_entities:
            count = text_lower.count(entity)
            entities.append({
                'text': entity,
                'type': 'PERSON',
                'count': count
            })
        
        # Expanded object and location patterns
        object_patterns = [
            # Weapons and tools
            r'\b(?:sword|blade|knife|dagger|hammer|axe|bow|arrow|spear|club|stick|rod|staff)\w*\b',
            # Clothing and accessories
            r'\b(?:dress|gown|cloak|coat|hat|bonnet|shoe|boot|ring|necklace|crown|tiara|belt|girdle)\w*\b',
            # Food and drink
            r'\b(?:bread|cake|bannock|soup|porridge|milk|water|wine|ale|meat|fish|apple|berry)\w*\b',
            # Buildings and places
            r'\b(?:castle|palace|house|cottage|hut|tower|room|kitchen|chamber|hall|barn|mill|smithy|shop)\w*\b',
            r'\b(?:kingdom|village|town|city|forest|wood|mountain|hill|valley|river|lake|sea|bridge|road|path)\w*\b',
            # Animals
            r'\b(?:horse|cow|pig|sheep|goat|chicken|hen|cock|cat|dog|wolf|fox|bear|deer|bird|dragon)\w*\b',
            # Magic and fantasy items
            r'\b(?:magic|magical|enchanted|cursed|golden|silver|crystal|diamond|ruby|emerald|treasure|gold|jewel)\w*\b',
            # Household items
            r'\b(?:table|chair|bed|fire|pot|pan|bowl|cup|plate|spoon|fork|knife|candle|lamp|mirror|box|chest)\w*\b',
            # Abstract concepts
            r'\b(?:love|hate|fear|joy|sorrow|anger|wisdom|courage|strength|beauty|ugliness|magic|spell|curse|wish)\w*\b'
        ]
        
        object_entities = set()
        for pattern in object_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0).strip().lower()
                if len(entity_text) > 2:
                    object_entities.add(entity_text)
        
        # Add object entities
        for entity in object_entities:
            count = text_lower.count(entity)
            entities.append({
                'text': entity,
                'type': 'OBJECT',
                'count': count
            })
        
        # Extract key phrases (adjective + noun combinations)
        phrase_pattern = r'\b(?:beautiful|ugly|brave|cowardly|wise|foolish|kind|cruel|young|old|big|small|tall|short|fast|slow|strong|weak|rich|poor|happy|sad|angry|peaceful|dark|light|golden|silver|magic|enchanted)\s+\w+\b'
        phrase_matches = re.finditer(phrase_pattern, text, re.IGNORECASE)
        
        phrase_entities = set()
        for match in phrase_matches:
            phrase = match.group(0).strip().lower()
            if len(phrase) > 5:  # Only keep substantial phrases
                phrase_entities.add(phrase)
        
        # Add phrase entities
        for entity in phrase_entities:
            count = text_lower.count(entity)
            entities.append({
                'text': entity,
                'type': 'CONCEPT',
                'count': count
            })
        
        return entities
    
    def extract_simple_relationships(self, text: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships using simple patterns."""
        relationships = []
        
        # Action patterns
        action_patterns = [
            (r'(\w+)\s+(fought|defeated|killed|saved|helped|met|found)\s+(\w+)', 'action'),
            (r'(\w+)\s+(had|owned|carried|held)\s+(\w+)', 'possession'),
            (r'(\w+)\s+(lived in|went to|came from)\s+(\w+)', 'location'),
            (r'(\w+)\s+(was|became)\s+(\w+)', 'state')
        ]
        
        for pattern, rel_type in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject, predicate, obj = match.groups()
                if any(e.lower() in subject.lower() for e in entities) or \
                   any(e.lower() in obj.lower() for e in entities):
                    relationships.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'type': rel_type
                    })
        
        return relationships
    
    def build_knowledge_graph(self, stories: Dict[str, str]):
        """Build simple but effective knowledge graph."""
        self.logger.info("Building simple knowledge graph...")
        
        cache_file = os.path.join(self.cache_dir, "simple_kg.json")
        
        if os.path.exists(cache_file):
            self.logger.info("Loading cached knowledge graph...")
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                self.entities = cache_data.get('entities', {})
                self.relationships = cache_data.get('relationships', [])
                self.entity_contexts = defaultdict(list, cache_data.get('entity_contexts', {}))
                
                # Validate cache has actual data
                if len(self.entities) == 0 and len(self.relationships) == 0:
                    self.logger.warning("Cache is empty, rebuilding knowledge graph...")
                else:
                    self.logger.info(f"Loaded KG with {len(self.entities)} entities and {len(self.relationships)} relationships")
                    return
            except Exception as e:
                self.logger.warning(f"Cache loading failed: {e}, rebuilding...")
        
        self.stories = stories
        all_entity_names = set()
        
        # Process each story
        for story_name, story_text in stories.items():
            self.logger.info(f"Processing story: {story_name} ({len(story_text)} chars)")
            
            # Extract entities
            entities = self.extract_simple_entities(story_text)
            self.logger.info(f"Extracted {len(entities)} entities from {story_name}")
            
            # Store entities
            for entity in entities:
                entity_name = entity['text'].lower()
                all_entity_names.add(entity_name)
                
                if entity_name not in self.entities:
                    self.entities[entity_name] = {
                        'original_text': entity['text'],
                        'type': entity['type'],
                        'stories': [],
                        'contexts': []
                    }
                
                if story_name not in self.entities[entity_name]['stories']:
                    self.entities[entity_name]['stories'].append(story_name)
                
                # Store context where entity appears
                sentences = story_text.split('.')
                for sentence in sentences:
                    if entity['text'].lower() in sentence.lower():
                        self.entities[entity_name]['contexts'].append(sentence.strip())
                        self.entity_contexts[entity_name].append(sentence.strip())
            
            # Extract relationships
            entity_names = [e['text'] for e in entities]
            relationships = self.extract_simple_relationships(story_text, entity_names)
            self.logger.info(f"Extracted {len(relationships)} relationships from {story_name}")
            
            for rel in relationships:
                self.relationships.append({
                    'subject': rel['subject'].lower(),
                    'predicate': rel['predicate'],
                    'object': rel['object'].lower(),
                    'type': rel['type'],
                    'story': story_name
                })
        
        # Save cache
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_data = {
                'entities': self.entities,
                'relationships': self.relationships,
                'entity_contexts': dict(self.entity_contexts)
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.info("Saved knowledge graph to cache")
        except Exception as e:
            self.logger.warning(f"Cache saving failed: {e}")
        
        self.logger.info(f"Built KG with {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from knowledge graph."""
        if not self.entities and not self.relationships:
            self.logger.warning("Knowledge graph is empty!")
            return ""
        
        question_lower = question.lower()
        relevant_info = []
        
        # Find entities mentioned in question (exact and partial matches)
        mentioned_entities = []
        question_words = set(question_lower.split())
        
        # First pass: exact entity matches
        for entity_name in self.entities.keys():
            if entity_name in question_lower:
                mentioned_entities.append(entity_name)
        
        # Second pass: partial word matches if no exact matches
        if not mentioned_entities:
            for entity_name in self.entities.keys():
                entity_words = set(entity_name.split())
                if question_words.intersection(entity_words):
                    mentioned_entities.append(entity_name)
        
        # Third pass: semantic matches if still no matches
        if not mentioned_entities:
            semantic_matches = {
                'wife': ['queen', 'woman', 'mother', 'goodwife'],
                'husband': ['king', 'man', 'father', 'goodman'],
                'woman': ['wife', 'queen', 'mother', 'goodwife', 'lady'],
                'man': ['husband', 'king', 'father', 'goodman', 'lord'],
                'old': ['old man', 'old woman', 'goodman', 'goodwife'],
                'young': ['young man', 'young woman', 'prince', 'princess'],
                'animal': ['cow', 'horse', 'pig', 'sheep', 'chicken', 'hen', 'cat', 'dog'],
                'food': ['bread', 'cake', 'bannock', 'soup', 'porridge', 'milk'],
                'place': ['house', 'cottage', 'castle', 'kingdom', 'forest', 'village'],
                'escape': ['run', 'flee', 'away', 'chase'],
                'catch': ['grab', 'seize', 'take', 'hold'],
                'hair': ['golden hair', 'beautiful hair']
            }
            
            for word in question_words:
                if word in semantic_matches:
                    for semantic_entity in semantic_matches[word]:
                        if semantic_entity in self.entities:
                            mentioned_entities.append(semantic_entity)
        
        # Get information about mentioned entities
        added_contexts = set()
        for entity in mentioned_entities[:5]:  # Limit to avoid too much info
            entity_data = self.entities[entity]
            
            # Add entity type information
            if entity_data['type'] != 'CONCEPT':
                relevant_info.append(f"{entity_data['original_text']} is a {entity_data['type'].lower()}")
            
            # Add best contexts from stories (more comprehensive)
            if entity_data['contexts']:
                # Get multiple relevant contexts, not just the shortest
                contexts = [c for c in entity_data['contexts'] if len(c.strip()) > 15]
                if contexts:
                    # Sort by relevance to question
                    context_scores = []
                    for ctx in contexts:
                        ctx_lower = ctx.lower()
                        score = sum(1 for word in question_words if word in ctx_lower and len(word) > 2)
                        context_scores.append((ctx, score))
                    
                    # Get top 2-3 contexts
                    context_scores.sort(key=lambda x: x[1], reverse=True)
                    for ctx, score in context_scores[:3]:
                        if ctx not in added_contexts:
                            relevant_info.append(f"Context: {ctx.strip()}")
                            added_contexts.add(ctx)
        
        # Find relevant relationships
        for rel in self.relationships[:10]:  # Increased relationship limit
            rel_text = f"{rel['subject']} {rel['predicate']} {rel['object']}"
            if any(word in rel_text.lower() for word in question_words if len(word) > 2):
                relationship_desc = f"{rel['subject']} {rel['predicate']} {rel['object']}"
                if relationship_desc not in relevant_info:
                    relevant_info.append(relationship_desc)
        
        # If still no information, provide broader context
        if not relevant_info and self.entities:
            # Look for any entities that might be related to question keywords
            broad_matches = []
            for entity_name, entity_data in self.entities.items():
                entity_contexts = ' '.join(entity_data['contexts'][:2]).lower()
                overlap = sum(1 for word in question_words if word in entity_contexts and len(word) > 2)
                if overlap > 0:
                    broad_matches.append((entity_name, entity_data, overlap))
            
            # Sort by relevance and add top matches
            broad_matches.sort(key=lambda x: x[2], reverse=True)
            for entity_name, entity_data, score in broad_matches[:3]:
                if entity_data['contexts']:
                    best_context = entity_data['contexts'][0]
                    if best_context not in added_contexts:
                        relevant_info.append(f"Related: {best_context.strip()}")
                        added_contexts.add(best_context)
        
        context = ". ".join(relevant_info[:8])  # Increased context limit
        
        if context:
            self.logger.info(f"KG retrieved context: {len(context)} chars, {len(relevant_info)} pieces")
        else:
            self.logger.warning("No KG context found")
            # Last resort fallback with more comprehensive information
            if self.entities:
                sample_entities = list(self.entities.items())[:5]
                fallback_info = []
                for entity_name, entity_data in sample_entities:
                    if entity_data['contexts']:
                        fallback_info.append(entity_data['contexts'][0])
                context = ". ".join(fallback_info)
        
        return context