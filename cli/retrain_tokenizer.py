# retrain_tokenizer.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tokenizer import HFTokenizer
import json
import requests
from core.ai_tools import OpenAIKnowledgeExtractor

def generate_diverse_training_data():
    """Generate diverse training data for tokenizer"""
    print("📚 Generating diverse training data for tokenizer...")
    
    # Categories for diverse text
    categories = [
        "mathematics", "physics", "chemistry", "biology", "computer science",
        "programming", "artificial intelligence", "machine learning", "data science",
        "history", "philosophy", "psychology", "economics", "literature",
        "music", "art", "geography", "astronomy", "engineering", "medicine",
        "technology", "business", "politics", "sports", "entertainment"
    ]
    
    training_files = []
    
    for category in categories:
        print(f"📝 Generating data for: {category}")
        
        # Use OpenAI to generate diverse text
        extractor = OpenAIKnowledgeExtractor()
        extractor.extract([category], max_cost=1.0, batch_size=2, train_on_download=False)
        
        # Find the generated file
        for file in os.listdir('datasets'):
            if category.replace(' ', '_').lower() in file.lower() and file.endswith('.jsonl'):
                training_files.append(os.path.join('datasets', file))
                break
    
    return training_files

def retrain_tokenizer(vocab_size=250000):
    """Retrain tokenizer with larger vocab size"""
    print(f"🔤 Retraining tokenizer with vocab size: {vocab_size:,}")
    
    # Generate diverse training data
    training_files = generate_diverse_training_data()
    
    if not training_files:
        print("⚠️  No training files found. Using existing data...")
        training_files = ['datasets/knowledge.jsonl']
    
    print(f"📚 Training on {len(training_files)} files: {training_files}")
    
    # Initialize and train tokenizer
    tokenizer = HFTokenizer(vocab_size=vocab_size)
    
    try:
        final_vocab_size = tokenizer.train(training_files, "models/tokenizer.json")
        print(f"✅ Tokenizer retrained successfully!")
        print(f"📊 Final vocab size: {final_vocab_size:,}")
        
        # Test the tokenizer
        test_text = "Hello world! This is a test of the new tokenizer with a larger vocabulary."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"🧪 Test encoding/decoding:")
        print(f"   Original: {test_text}")
        print(f"   Tokens: {tokens[:10]}...")
        print(f"   Decoded: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error retraining tokenizer: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting tokenizer retraining process...")
    
    # Retrain with 250K vocab
    success = retrain_tokenizer(vocab_size=250000)
    
    if success:
        print("\n🎉 Tokenizer retraining completed successfully!")
        print("📊 New vocab size: 250,000")
        print("💾 Saved to: models/tokenizer.json")
    else:
        print("\n❌ Tokenizer retraining failed!")

if __name__ == '__main__':
    main() 