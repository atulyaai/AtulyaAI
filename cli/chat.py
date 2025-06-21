# chat.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model import MultimodalModel
import json
import time
from datetime import datetime

class ChatInterface:
    def __init__(self):
        self.model_path = 'models/model.pt'
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def load_model(self):
        """Load the trained model"""
        try:
            print("🔄 Loading model...")
            self.model = MultimodalModel()
            
            if os.path.exists(self.model_path):
                self.model.load(self.model_path)
                print("✅ Model loaded successfully!")
                
                # Get model stats
                stats = self.model.get_parameter_stats()
                print(f"📊 Model Stats:")
                print(f"   Parameters: {stats['total_parameters']:,}")
                print(f"   Model Size: {stats['model_size_mb']:.2f} MB")
                print(f"   Vocab Size: {stats['vocab_size']:,}")
                
                self.tokenizer = self.model.tokenizer
                return True
            else:
                print("❌ Model file not found!")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_response(self, user_input, max_length=100):
        """Generate response using the model"""
        try:
            # Prepare input
            audio = torch.zeros(1, 1, 128, 128)  # Dummy audio
            video = torch.zeros(1, 3, 16, 64, 64)  # Dummy video
            
            # Get model output
            with torch.no_grad():
                output = self.model(user_input, audio, video)
            
            # Extract text features
            if 'text_feat' in output:
                logits = output['text_feat']
                
                # Generate response using greedy decoding
                response_tokens = []
                current_input = torch.tensor([self.tokenizer.encode(user_input)], dtype=torch.long)
                
                for _ in range(max_length):
                    # Get next token
                    with torch.no_grad():
                        output = self.model(current_input, audio, video)
                        logits = output['text_feat']
                        next_token_logits = logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Add to response
                    response_tokens.append(next_token.item())
                    
                    # Update input
                    current_input = torch.cat([current_input, next_token], dim=1)
                    
                    # Stop if end token
                    if next_token.item() == 2:  # </s> token
                        break
                
                # Decode response
                response = self.tokenizer.decode(response_tokens)
                return response
            
            return "Sorry, I couldn't generate a response."
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def chat(self):
        """Main chat loop"""
        print("\n" + "="*60)
        print("🤖 ATULYA AI CHAT INTERFACE")
        print("="*60)
        print("Type 'quit' to exit, 'help' for commands")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("🤖 Atulya: ", end="", flush=True)
                start_time = time.time()
                
                response = self.generate_response(user_input)
                
                generation_time = time.time() - start_time
                print(response)
                print(f"⏱️  Generated in {generation_time:.2f}s")
                
                # Save to history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'assistant': response,
                    'generation_time': generation_time
                })
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help commands"""
        print("\n📖 Available Commands:")
        print("  help     - Show this help")
        print("  stats    - Show model statistics")
        print("  history  - Show conversation history")
        print("  quit     - Exit chat")
        print("  exit     - Exit chat")
        print("  q        - Exit chat")
    
    def show_stats(self):
        """Show model statistics"""
        if self.model:
            stats = self.model.get_parameter_stats()
            print(f"\n📊 Model Statistics:")
            print(f"   Total Parameters: {stats['total_parameters']:,}")
            print(f"   Trainable Parameters: {stats['trainable_parameters']:,}")
            print(f"   Model Size: {stats['model_size_mb']:.2f} MB")
            print(f"   Vocab Size: {stats['vocab_size']:,}")
            print(f"   Text Model Params: {stats['text_model_params']:,}")
            print(f"   Audio Model Params: {stats['audio_model_params']:,}")
            print(f"   Video Model Params: {stats['video_model_params']:,}")
        else:
            print("❌ Model not loaded!")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} exchanges):")
        for i, exchange in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            print(f"\n{i}. {exchange['timestamp']}")
            print(f"   👤 You: {exchange['user'][:50]}...")
            print(f"   🤖 Atulya: {exchange['assistant'][:50]}...")
            print(f"   ⏱️  Time: {exchange['generation_time']:.2f}s")

def main():
    """Main function"""
    chat = ChatInterface()
    
    if chat.load_model():
        chat.chat()
    else:
        print("❌ Failed to load model. Please ensure model.pt exists.")

if __name__ == '__main__':
    main() 