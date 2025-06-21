# dynamic_topics.py
import json
import os
import requests
from datetime import datetime
import random

class DynamicTopicGenerator:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.topics_file = 'datasets/dynamic_topics.json'
        self.categories_file = 'datasets/categories.json'
        self.endpoint = 'https://api.openai.com/v1/chat/completions'
        
    def _get_api_key(self):
        config_path = 'configs/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'openai_api_key' in config and config['openai_api_key']:
                return config['openai_api_key']
        raise RuntimeError('OpenAI API key not found in config.')
    
    def expand_categories(self, max_categories=50):
        """Dynamically expand categories using OpenAI"""
        print(f"🌍 Expanding categories dynamically...")
        
        # Start with basic categories
        base_categories = [
            "mathematics", "physics", "chemistry", "biology", "computer science",
            "programming", "artificial intelligence", "machine learning", "data science"
        ]
        
        # Load existing categories
        existing_categories = self._load_categories()
        all_categories = list(set(base_categories + existing_categories))
        
        # Generate new categories
        new_categories = self._generate_new_categories(all_categories, max_categories - len(all_categories))
        all_categories.extend(new_categories)
        
        # Save categories
        self._save_categories(all_categories)
        return all_categories
    
    def _generate_new_categories(self, existing_categories, count):
        """Generate new categories using OpenAI"""
        prompt = f"""Generate {count} new academic and professional categories that are NOT in this list: {existing_categories[:10]}
        
        Focus on:
        - Emerging fields and technologies
        - Interdisciplinary areas
        - Specialized domains
        - Cultural and social sciences
        - Applied sciences and engineering
        
        Return only a JSON array of category strings."""
        
        return self._query_openai_for_list(prompt, count)
    
    def generate_topics_for_category(self, category, count=20, difficulty="mixed"):
        """Generate topics for a specific category with dynamic difficulty"""
        prompt = f"""Generate {count} diverse topics for {category} covering:
        
        - Basic/Fundamental (30%): Core concepts, principles, introductory material
        - Intermediate/Applied (40%): Practical applications, problem-solving, deeper understanding
        - Advanced/Research (30%): Cutting-edge concepts, specialized knowledge, emerging trends
        
        Make topics specific and actionable for learning.
        Return only a JSON array of topic strings."""
        
        return self._query_openai_for_list(prompt, count)
    
    def generate_all_topics(self, max_topics=1000, topics_per_category=20):
        """Generate topics for all categories dynamically"""
        print(f"🎯 Generating {max_topics} topics across all categories...")
        
        # Expand categories first
        categories = self.expand_categories()
        print(f"📊 Using {len(categories)} categories")
        
        all_topics = []
        topics_per_cat = min(topics_per_category, max_topics // len(categories))
        
        for category in categories:
            if len(all_topics) >= max_topics:
                break
                
            print(f"📝 Generating topics for: {category}")
            category_topics = self.generate_topics_for_category(category, topics_per_cat)
            all_topics.extend(category_topics)
            print(f"✅ Generated {len(category_topics)} topics for {category}")
        
        # Save topics with metadata
        self._save_topics(all_topics, categories)
        return all_topics
    
    def _query_openai_for_list(self, prompt, expected_count):
        """Query OpenAI API for list generation"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 2000,
            'temperature': 0.8
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    items = json.loads(content)
                    if isinstance(items, list):
                        return items[:expected_count]
                except json.JSONDecodeError:
                    # Fallback parsing
                    lines = content.split('\n')
                    items = []
                    for line in lines:
                        line = line.strip().replace('"', '').replace(',', '').replace('[', '').replace(']', '')
                        if line and len(line) > 3 and not line.startswith('-'):
                            items.append(line)
                    return items[:expected_count]
            
            return []
            
        except Exception as e:
            print(f"❌ Error querying OpenAI: {e}")
            return []
    
    def _load_categories(self):
        """Load existing categories"""
        if os.path.exists(self.categories_file):
            try:
                with open(self.categories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('categories', [])
            except Exception as e:
                print(f"⚠️  Error loading categories: {e}")
        return []
    
    def _save_categories(self, categories):
        """Save categories to file"""
        os.makedirs('datasets', exist_ok=True)
        
        data = {
            'categories': categories,
            'total_count': len(categories),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.categories_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(categories)} categories to {self.categories_file}")
    
    def _save_topics(self, topics, categories):
        """Save topics with metadata"""
        os.makedirs('datasets', exist_ok=True)
        
        data = {
            'topics': topics,
            'categories': categories,
            'total_count': len(topics),
            'category_count': len(categories),
            'last_updated': datetime.now().isoformat(),
            'metadata': {
                'generation_method': 'dynamic_openai',
                'difficulty_distribution': 'mixed',
                'avg_topics_per_category': len(topics) // len(categories) if categories else 0
            }
        }
        
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(topics)} topics to {self.topics_file}")
    
    def get_topics_by_difficulty(self, difficulty="all"):
        """Get topics filtered by difficulty"""
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                topics = data.get('topics', [])
                
                if difficulty == "all":
                    return topics
                elif difficulty == "basic":
                    return topics[:len(topics)//3]
                elif difficulty == "intermediate":
                    return topics[len(topics)//3:2*len(topics)//3]
                elif difficulty == "advanced":
                    return topics[2*len(topics)//3:]
        
        return []

def main():
    """Test dynamic topic generation"""
    generator = DynamicTopicGenerator()
    
    # Generate topics dynamically
    topics = generator.generate_all_topics(max_topics=200, topics_per_category=10)
    
    print(f"\n{'='*60}")
    print(f"🎉 DYNAMIC TOPIC GENERATION COMPLETED!")
    print(f"{'='*60}")
    print(f"✅ Total topics: {len(topics)}")
    print(f"📊 Categories: {len(generator._load_categories())}")
    print(f"💾 Saved to: {generator.topics_file}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main() 