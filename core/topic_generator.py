# topic_generator.py
import json
import os
import requests
from datetime import datetime

class TopicGenerator:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.topics_file = 'datasets/topics.json'
        self.trained_file = 'logs/trained_topics.txt'
        self.endpoint = 'https://api.openai.com/v1/chat/completions'
        
    def _get_api_key(self):
        config_path = 'configs/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'openai_api_key' in config and config['openai_api_key']:
                return config['openai_api_key']
        raise RuntimeError('OpenAI API key not found in config.')
    
    def generate_topics(self, category="all", max_topics=100, difficulty="progressive"):
        """Generate topics from basic to advanced"""
        
        # Load existing topics
        existing_topics = self._load_existing_topics()
        trained_topics = self._load_trained_topics()
        
        print(f"🎯 Generating {max_topics} topics for category: {category}")
        print(f"📊 Existing topics: {len(existing_topics)}")
        print(f"✅ Trained topics: {len(trained_topics)}")
        
        # Generate new topics
        new_topics = []
        
        if difficulty == "progressive":
            # Generate basic topics first
            basic_topics = self._generate_basic_topics(category, max_topics // 3)
            new_topics.extend(basic_topics)
            
            # Generate intermediate topics
            intermediate_topics = self._generate_intermediate_topics(category, max_topics // 3)
            new_topics.extend(intermediate_topics)
            
            # Generate advanced topics
            advanced_topics = self._generate_advanced_topics(category, max_topics // 3)
            new_topics.extend(advanced_topics)
        else:
            # Generate all topics at once
            all_topics = self._generate_all_topics(category, max_topics)
            new_topics.extend(all_topics)
        
        # Remove duplicates and already trained topics
        unique_topics = []
        for topic in new_topics:
            if topic not in existing_topics and topic not in trained_topics:
                unique_topics.append(topic)
        
        # Save new topics
        all_topics = existing_topics + unique_topics
        self._save_topics(all_topics)
        
        print(f"✅ Generated {len(unique_topics)} new unique topics")
        print(f"📊 Total topics available: {len(all_topics)}")
        
        return unique_topics
    
    def _generate_basic_topics(self, category, count):
        """Generate basic/fundamental topics"""
        prompt = f"""Generate {count} basic and fundamental topics for {category} that a beginner should learn first. 
        Focus on foundational concepts, basic principles, and essential knowledge.
        Return only a JSON array of topic strings, no explanations.
        Examples for different categories:
        - Math: ["basic arithmetic", "fractions", "decimals", "percentages"]
        - Science: ["scientific method", "basic chemistry", "simple physics", "biology basics"]
        - Programming: ["variables", "loops", "functions", "basic data types"]
        - Language: ["basic grammar", "vocabulary", "pronunciation", "simple sentences"]"""
        
        return self._query_openai_for_topics(prompt, count)
    
    def _generate_intermediate_topics(self, category, count):
        """Generate intermediate topics"""
        prompt = f"""Generate {count} intermediate topics for {category} that build upon basic knowledge.
        Focus on practical applications, problem-solving, and deeper understanding.
        Return only a JSON array of topic strings, no explanations.
        Examples:
        - Math: ["algebra", "geometry", "statistics", "calculus basics"]
        - Science: ["organic chemistry", "thermodynamics", "genetics", "ecology"]
        - Programming: ["object-oriented programming", "algorithms", "data structures", "web development"]
        - Language: ["complex grammar", "idioms", "writing skills", "conversation"]"""
        
        return self._query_openai_for_topics(prompt, count)
    
    def _generate_advanced_topics(self, category, count):
        """Generate advanced/expert topics"""
        prompt = f"""Generate {count} advanced and expert-level topics for {category} that require deep knowledge.
        Focus on cutting-edge concepts, research areas, and specialized applications.
        Return only a JSON array of topic strings, no explanations.
        Examples:
        - Math: ["abstract algebra", "topology", "number theory", "mathematical logic"]
        - Science: ["quantum mechanics", "molecular biology", "astrophysics", "neuroscience"]
        - Programming: ["machine learning", "distributed systems", "compiler design", "quantum computing"]
        - Language: ["linguistics", "translation theory", "creative writing", "rhetoric"]"""
        
        return self._query_openai_for_topics(prompt, count)
    
    def _generate_all_topics(self, category, count):
        """Generate a mix of all difficulty levels"""
        prompt = f"""Generate {count} diverse topics for {category} covering basic, intermediate, and advanced levels.
        Mix foundational concepts with practical applications and cutting-edge areas.
        Return only a JSON array of topic strings, no explanations.
        Ensure variety in difficulty and scope."""
        
        return self._query_openai_for_topics(prompt, count)
    
    def _query_openai_for_topics(self, prompt, expected_count):
        """Query OpenAI API for topic generation"""
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
                
                # Try to parse JSON array
                try:
                    topics = json.loads(content)
                    if isinstance(topics, list):
                        return topics[:expected_count]
                except json.JSONDecodeError:
                    # Fallback: extract topics from text
                    lines = content.split('\n')
                    topics = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-'):
                            # Clean up the line
                            topic = line.replace('"', '').replace(',', '').strip()
                            if topic and len(topic) > 3:
                                topics.append(topic)
                    return topics[:expected_count]
            
            print(f"⚠️  API request failed: {response.status_code}")
            return []
            
        except Exception as e:
            print(f"❌ Error querying OpenAI: {e}")
            return []
    
    def _load_existing_topics(self):
        """Load existing topics from file"""
        if os.path.exists(self.topics_file):
            try:
                with open(self.topics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('topics', [])
            except Exception as e:
                print(f"⚠️  Error loading topics: {e}")
        return []
    
    def _load_trained_topics(self):
        """Load already trained topics"""
        if os.path.exists(self.trained_file):
            try:
                with open(self.trained_file, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"⚠️  Error loading trained topics: {e}")
        return set()
    
    def _save_topics(self, topics):
        """Save topics to file"""
        os.makedirs(os.path.dirname(self.topics_file), exist_ok=True)
        
        data = {
            'topics': topics,
            'total_count': len(topics),
            'last_updated': datetime.now().isoformat(),
            'categories': ['math', 'science', 'programming', 'language', 'history', 'philosophy', 'arts', 'technology']
        }
        
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Topics saved to {self.topics_file}")
    
    def get_untrained_topics(self, limit=None):
        """Get topics that haven't been trained yet"""
        existing_topics = self._load_existing_topics()
        trained_topics = self._load_trained_topics()
        
        untrained = [topic for topic in existing_topics if topic not in trained_topics]
        
        if limit:
            untrained = untrained[:limit]
        
        return untrained

def main():
    """Main function to generate topics"""
    generator = TopicGenerator()
    
    # Generate topics for different categories
    categories = ['math', 'science', 'programming', 'language', 'history', 'philosophy', 'arts', 'technology']
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"🎯 Generating topics for: {category.upper()}")
        print(f"{'='*60}")
        
        new_topics = generator.generate_topics(
            category=category,
            max_topics=50,  # 50 topics per category
            difficulty="progressive"
        )
        
        print(f"✅ Generated {len(new_topics)} new topics for {category}")
    
    # Show summary
    all_topics = generator._load_existing_topics()
    trained_topics = generator._load_trained_topics()
    untrained_topics = generator.get_untrained_topics()
    
    print(f"\n{'='*60}")
    print(f"📊 TOPIC GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total topics available: {len(all_topics)}")
    print(f"Already trained: {len(trained_topics)}")
    print(f"Ready for training: {len(untrained_topics)}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main() 