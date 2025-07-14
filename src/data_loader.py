from datasets import load_dataset
import pandas as pd
import json
import os
from typing import Dict, List, Any
import logging

class FairytaleQALoader:
    """Data loader for FairytaleQA dataset with caching and preprocessing."""
    
    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_test_split(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load test split of FairytaleQA dataset."""
        cache_file = os.path.join(self.cache_dir, f"fairytale_qa_test_{limit}.json")
        
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached test data from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        self.logger.info("Downloading FairytaleQA dataset...")
        dataset = load_dataset("WorkInTheDark/FairytaleQA")
        test_data = dataset['test']
        
        # Convert to list and limit
        processed_data = []
        for i, item in enumerate(test_data):
            if i >= limit:
                break
                
            processed_item = {
                'id': i,
                'story_name': item['story_name'],
                'story_section': item['story_section'],
                'question': item['question'],
                'answer': item['answer1'],  # Use primary answer
                'answer2': item.get('answer2', ''),
                'local_or_sum': item['local-or-sum'],
                'attribute': item['attribute'],
                'ex_or_im': item['ex-or-im'],
                'ex_or_im2': item.get('ex-or-im2', '')
            }
            processed_data.append(processed_item)
        
        # Cache the processed data
        with open(cache_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        self.logger.info(f"Loaded {len(processed_data)} test questions")
        return processed_data
    
    def get_unique_stories(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract unique stories from the dataset."""
        stories = {}
        for item in data:
            story_name = item['story_name']
            if story_name not in stories:
                stories[story_name] = ""
            
            # Concatenate story sections
            if item['story_section'] not in stories[story_name]:
                stories[story_name] += " " + item['story_section']
        
        # Clean up stories
        for story_name in stories:
            stories[story_name] = stories[story_name].strip()
        
        self.logger.info(f"Extracted {len(stories)} unique stories")
        return stories
    
    def get_story_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        df = pd.DataFrame(data)
        
        stats = {
            'total_questions': len(data),
            'unique_stories': df['story_name'].nunique(),
            'attribute_distribution': df['attribute'].value_counts().to_dict(),
            'local_vs_summary': df['local_or_sum'].value_counts().to_dict(),
            'explicit_vs_implicit': df['ex_or_im'].value_counts().to_dict(),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
            'avg_story_length': df['story_section'].str.len().mean()
        }
        
        return stats