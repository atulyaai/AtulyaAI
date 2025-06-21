import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.processors import TemplateProcessing

class HFTokenizer:
    def __init__(self, vocab_size=250000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        print(f"🔤 Initializing tokenizer with vocab size: {vocab_size:,}")
        
        # Force vocab size to be large
        if vocab_size < 100000:
            print(f"⚠️  Warning: Small vocab size {vocab_size}. Consider using 250K+ for better performance.")
        
    def train(self, files, save_path="models/tokenizer.json"):
        """Train the tokenizer on files"""
        print(f"🚀 Training tokenizer on {len(files)} files with vocab size {self.vocab_size:,}")
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
        # Configure decoder
        self.tokenizer.decoder = decoders.ByteLevel()
        
        # Configure post-processor
        self.tokenizer.post_processor = TemplateProcessing(
            single="$A",
            pair="$A:0 $B:1",
            special_tokens=[
                ("<s>", 0),
                ("<pad>", 1),
                ("</s>", 2),
                ("<unk>", 3),
            ],
        )
        
        # Configure trainer with larger vocab
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=2  # Lower frequency for larger vocab
        )
        
        # Train
        print(f"📚 Training on files: {files}")
        self.tokenizer.train(files, trainer)
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.tokenizer.save(save_path)
        print(f"✅ Tokenizer saved to {save_path}")
        print(f"📊 Final vocab size: {self.tokenizer.get_vocab_size():,}")
        
        # Warn if vocab is too small
        final_size = self.tokenizer.get_vocab_size()
        if final_size < 50000:
            print(f"⚠️  WARNING: Final vocab size {final_size:,} is much smaller than requested {self.vocab_size:,}")
            print(f"   This may indicate insufficient training data or need for more diverse text.")
        
        return final_size
        
    def load(self, path):
        """Load tokenizer from file"""
        if os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
            vocab_size = self.tokenizer.get_vocab_size()
            print(f"📖 Loaded tokenizer from {path}")
            print(f"📊 Vocab size: {vocab_size:,}")
            self.vocab_size = vocab_size
        else:
            print(f"⚠️  Tokenizer file not found: {path}")
            
    def encode(self, text):
        """Encode text to token IDs"""
        if self.tokenizer is None:
            print("⚠️  Tokenizer not loaded. Returning character codes.")
            return [ord(c) for c in text]
        
        encoding = self.tokenizer.encode(text)
        return encoding.ids
        
    def decode(self, ids):
        """Decode token IDs to text"""
        if self.tokenizer is None:
            print("⚠️  Tokenizer not loaded. Returning character codes.")
            return ''.join([chr(id) if id < 65536 else '?' for id in ids])
        
        return self.tokenizer.decode(ids) 