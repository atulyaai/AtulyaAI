# auto_train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import time
from datetime import datetime
from models.model import MultimodalModel
from core.training import train_on_topic
from core.ai_tools import OpenAIKnowledgeExtractor

class AutoTrainer:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.topics_file = 'datasets/topics.json'
        self.trained_file = 'logs/trained_topics.txt'
        self.model_path = 'models/model.pt'
        
    def _get_api_key(self):
        config_path = 'configs/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'openai_api_key' in config and config['openai_api_key']:
                return config['openai_api_key']
        raise RuntimeError('OpenAI API key not found in config.')
    
    def generate_topics(self, max_topics=200):
        """Generate topics using OpenAI API"""
        print(f"🎯 Generating {max_topics} topics using OpenAI...")
        
        # Basic topic categories
        categories = [
            "mathematics", "physics", "chemistry", "biology", "computer science",
            "programming", "artificial intelligence", "machine learning", "data science",
            "history", "philosophy", "psychology", "economics", "literature",
            "music", "art", "geography", "astronomy", "engineering", "medicine"
        ]
        
        all_topics = []
        
        for category in categories:
            topics_per_category = max_topics // len(categories)
            category_topics = self._generate_category_topics(category, topics_per_category)
            all_topics.extend(category_topics)
            print(f"✅ Generated {len(category_topics)} topics for {category}")
        
        # Save topics
        self._save_topics(all_topics)
        return all_topics
    
    def _generate_category_topics(self, category, count):
        """Generate topics for a specific category"""
        prompt = f"""Generate {count} diverse topics for {category} covering basic to advanced levels.
        Include foundational concepts, practical applications, and cutting-edge areas.
        Return only a JSON array of topic strings.
        Examples for {category}:
        - Basic: fundamental concepts, principles, introductory topics
        - Intermediate: applications, problem-solving, deeper understanding  
        - Advanced: research areas, specialized knowledge, emerging trends"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1500,
            'temperature': 0.8
        }
        
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', 
                                   headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    topics = json.loads(content)
                    if isinstance(topics, list):
                        return topics[:count]
                except:
                    # Fallback parsing
                    lines = content.split('\n')
                    topics = []
                    for line in lines:
                        line = line.strip().replace('"', '').replace(',', '')
                        if line and len(line) > 3:
                            topics.append(line)
                    return topics[:count]
            
            return []
            
        except Exception as e:
            print(f"❌ Error generating topics for {category}: {e}")
            return []
    
    def _save_topics(self, topics):
        """Save topics to file"""
        os.makedirs('datasets', exist_ok=True)
        
        data = {
            'topics': topics,
            'total_count': len(topics),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(topics)} topics to {self.topics_file}")
    
    def get_untrained_topics(self, limit=None):
        """Get topics that haven't been trained yet"""
        # Load existing topics
        if os.path.exists(self.topics_file):
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_topics = data.get('topics', [])
        else:
            all_topics = []
        
        # Load trained topics
        trained_topics = set()
        if os.path.exists(self.trained_file):
            with open(self.trained_file, 'r', encoding='utf-8') as f:
                trained_topics = set(line.strip() for line in f if line.strip())
        
        # Get untrained topics
        untrained = [topic for topic in all_topics if topic not in trained_topics]
        
        if limit:
            untrained = untrained[:limit]
        
        return untrained
    
    def run_automated_training(self, max_topics=200, topics_per_batch=10):
        """Run automated training on generated topics"""
        print(f"🚀 Starting automated training process...")
        print(f"📊 Target: {max_topics} topics, {topics_per_batch} per batch")
        
        # Generate topics if needed
        untrained_topics = self.get_untrained_topics()
        if len(untrained_topics) < max_topics:
            print(f"📝 Generating more topics...")
            new_topics = self.generate_topics(max_topics - len(untrained_topics))
            untrained_topics = self.get_untrained_topics(max_topics)
        
        print(f"🎯 Found {len(untrained_topics)} topics ready for training")
        
        # Load or create model
        if os.path.exists(self.model_path):
            print(f"📖 Loading existing model from {self.model_path}")
            model = MultimodalModel()
            model.load(self.model_path)
        else:
            print(f"🆕 Creating new model")
            model = MultimodalModel()
        
        # Train on topics in batches
        total_trained = 0
        batch_count = 0
        
        for i in range(0, len(untrained_topics), topics_per_batch):
            batch_topics = untrained_topics[i:i + topics_per_batch]
            batch_count += 1
            
            print(f"\n{'='*80}")
            print(f"📦 BATCH {batch_count}: Training on {len(batch_topics)} topics")
            print(f"{'='*80}")
            
            for j, topic in enumerate(batch_topics):
                try:
                    print(f"\n🎯 [{i+j+1}/{len(untrained_topics)}] Training on: {topic}")
                    
                    # Train on this topic
                    result = train_on_topic(model, topic)
                    
                    if result:
                        total_trained += 1
                        print(f"✅ Successfully trained on: {topic}")
                    else:
                        print(f"⚠️  Skipped training on: {topic}")
                    
                    # Small delay between topics
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error training on {topic}: {e}")
                    continue
            
            # Save model after each batch
            model.save(self.model_path)
            print(f"💾 Model saved after batch {batch_count}")
            
            # Progress summary
            print(f"\n📊 PROGRESS: {total_trained}/{len(untrained_topics)} topics trained")
            
            # Optional: pause between batches
            if batch_count < len(untrained_topics) // topics_per_batch:
                print(f"⏸️  Pausing 5 seconds before next batch...")
                time.sleep(5)
        
        print(f"\n{'='*80}")
        print(f"🎉 AUTOMATED TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"✅ Total topics trained: {total_trained}")
        print(f"📊 Model saved to: {self.model_path}")
        print(f"📝 Training log: logs/topic_training_log.jsonl")
        print(f"{'='*80}")
        
        return total_trained

def main():
    """Main function"""
    trainer = AutoTrainer()
    
    # Run automated training
    trainer.run_automated_training(
        max_topics=100,  # Start with 100 topics
        topics_per_batch=5  # Train 5 topics at a time
    )

if __name__ == '__main__':
    main() 