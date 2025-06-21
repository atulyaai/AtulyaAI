# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from models.modalities import CustomTextModel, CustomAudioModel, CustomVideoModel, SentimentModel, EmpathyModel, ToneModel, FusionModel
from models.tokenizer import HFTokenizer
from core.utilities import get_enabled_features

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=32):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_size)
        self.up = nn.Linear(adapter_size, hidden_size)
        self.act = nn.ReLU()
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class DynamicBlock(nn.Module):
    def __init__(self, hidden_size, adapter_size=32, use_adapter=False, top_k_neurons=None):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.use_adapter = use_adapter
        self.adapter = Adapter(hidden_size, adapter_size) if use_adapter else None
        self.top_k_neurons = top_k_neurons
        self.act = nn.ReLU()
    def forward(self, x):
        out = self.linear1(x)
        if self.top_k_neurons:
            # Sparse activation: only top-k neurons
            values, idx = torch.topk(out, self.top_k_neurons, dim=-1)
            mask = torch.zeros_like(out).scatter_(-1, idx, 1.0)
            out = out * mask
        out = self.act(out)
        out = self.linear2(out)
        if self.use_adapter and self.adapter:
            out = self.adapter(out)
        return out

class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_blocks, top_k_blocks=1):
        super().__init__()
        self.gate = nn.Linear(input_size, num_blocks)
        self.top_k_blocks = top_k_blocks
    def forward(self, x):
        # x: [batch, hidden_size]
        logits = self.gate(x)
        topk = torch.topk(logits, self.top_k_blocks, dim=-1)
        mask = torch.zeros_like(logits).scatter_(-1, topk.indices, 1.0)
        return mask

class DynamicMoEModel(nn.Module):
    def __init__(self, vocab_size=250000, hidden_size=256, num_layers=4, num_experts=4, adapter_size=32, top_k_blocks=1, top_k_neurons=None, use_adapter=False, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.adapter_size = adapter_size
        self.top_k_blocks = top_k_blocks
        self.top_k_neurons = top_k_neurons
        self.use_adapter = use_adapter
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Create experts (blocks)
        self.blocks = nn.ModuleList([
            DynamicBlock(hidden_size, adapter_size, use_adapter, top_k_neurons)
            for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(hidden_size, num_experts, top_k_blocks)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, *args, **kwargs):
        # x: [batch, seq_len]
        emb = self.embedding(x)
        # For each position, route through top-k experts
        batch, seq, _ = emb.shape
        out = emb
        for _ in range(self.num_layers):
            # Pool to get a routing signal (mean over seq)
            pooled = out.mean(dim=1)
            mask = self.gating(pooled)  # [batch, num_experts]
            expert_outs = []
            for i, block in enumerate(self.blocks):
                # Only compute for active experts
                active = mask[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                block_out = block(out)
                expert_outs.append(block_out * active)
            out = sum(expert_outs)
        logits = self.fc_out(out)
        return logits
    def grow(self, add_experts=1, add_layers=0):
        # Add new experts (blocks)
        for _ in range(add_experts):
            self.blocks.append(DynamicBlock(self.hidden_size, self.adapter_size, self.use_adapter, self.top_k_neurons))
            self.num_experts += 1
        self.gating = GatingNetwork(self.hidden_size, self.num_experts, self.top_k_blocks)
        self.num_layers += add_layers
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    def active_parameters(self, batch_size=1):
        # Estimate: params in embedding, gating, fc_out, and top_k_blocks * block params * num_layers
        emb = self.embedding.weight.numel()
        gate = sum(p.numel() for p in self.gating.parameters())
        fc = self.fc_out.weight.numel() + self.fc_out.bias.numel()
        block = sum(p.numel() for p in self.blocks[0].parameters())
        return emb + gate + fc + (block * self.top_k_blocks * self.num_layers)
    def summary(self):
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_experts': self.num_experts,
            'adapter_size': self.adapter_size,
            'top_k_blocks': self.top_k_blocks,
            'top_k_neurons': self.top_k_neurons,
            'use_adapter': self.use_adapter,
            'total_parameters': self.total_parameters(),
            'active_parameters': self.active_parameters(),
        }
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

class MultimodalModel(nn.Module):
    def __init__(self, vocab_size=250000):
        super().__init__()
        features = get_enabled_features()
        self.tokenizer = HFTokenizer(vocab_size)
        # Load tokenizer if exists
        if os.path.exists("models/tokenizer.json"):
            try:
                self.tokenizer.load("models/tokenizer.json")
                print("✅ Tokenizer loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load tokenizer: {e}")
        
        self.text_model = CustomTextModel(vocab_size) if features.get('text', True) else None
        self.audio_model = CustomAudioModel() if features.get('audio', False) else None
        self.video_model = CustomVideoModel() if features.get('video', False) else None
        self.sentiment_model = SentimentModel() if features.get('sentiment', False) else None
        self.empathy_model = EmpathyModel() if features.get('empathy', False) else None
        self.tone_model = ToneModel() if features.get('tone', False) else None
        self.fusion_model = FusionModel()
        self.weights = None

    def get_parameter_stats(self):
        """Get comprehensive parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024*1024)  # float32
        
        # Breakdown by component
        text_params = sum(p.numel() for p in self.text_model.parameters()) if self.text_model else 0
        audio_params = sum(p.numel() for p in self.audio_model.parameters()) if self.audio_model else 0
        video_params = sum(p.numel() for p in self.video_model.parameters()) if self.video_model else 0
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'text_model_params': text_params,
            'audio_model_params': audio_params,
            'video_model_params': video_params,
            'vocab_size': self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 250000
        }

    def tensor_to_str(self, tensor):
        # Convert a tensor of char codes to a string
        if tensor.dim() > 1:
            tensor = tensor[0]
        return ''.join([chr(int(c)) for c in tensor if c > 0])

    def forward(self, text, audio, video, y=None):
        result = {}
        if self.text_model:
            # If text is a batch tensor, decode each to string and encode
            if isinstance(text, torch.Tensor) and text.dim() > 1:
                text_strs = [self.tensor_to_str(t) for t in text]
                input_ids = [torch.tensor(self.tokenizer.encode(s), dtype=torch.long) for s in text_strs]
                # Pad input_ids to the same length as y if provided
                from torch.nn.utils.rnn import pad_sequence
                if y is not None:
                    max_len = y.size(1)
                    input_ids = [torch.cat([ids, torch.zeros(max_len - ids.size(0), dtype=ids.dtype)]) if ids.size(0) < max_len else ids[:max_len] for ids in input_ids]
                    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
                else:
                    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            else:
                text_str = self.tensor_to_str(text) if isinstance(text, torch.Tensor) else text
                input_ids = torch.tensor([self.tokenizer.encode(text_str)], dtype=torch.long)
            text_vec = self.text_model(input_ids)
            result['text_feat'] = text_vec
        else:
            text_vec = None
        if self.audio_model:
            audio_feat = self.audio_model(audio)
            result['audio_feat'] = audio_feat
        else:
            audio_feat = None
        if self.video_model:
            video_feat = self.video_model(video)
            result['video_feat'] = video_feat
        else:
            video_feat = None
        if self.sentiment_model and text_vec is not None:
            result['sentiment'] = self.sentiment_model.process(text_vec.mean(dim=1))
        if self.empathy_model and text_vec is not None:
            result['empathy'] = self.empathy_model.process(text_vec.mean(dim=1))
        if self.tone_model and audio_feat is not None:
            result['tone'] = self.tone_model.process(audio_feat)
        # Optionally fuse features
        result['fused'] = self.fusion_model.fuse(result)
        return result

    def save(self, path="models/model.pt"):
        """Save model with proper error handling and directory creation"""
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            state_dict = self.state_dict()
            torch.save(state_dict, path)
            file_size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ Model saved: {path} ({file_size_mb:.2f} MB)")
            
            # Save model info
            info_path = path.replace('.pt', '_info.json')
            model_info = {
                'parameter_stats': self.get_parameter_stats(),
                'model_type': 'MultimodalModel',
                'save_timestamp': str(torch.cuda.Event() if torch.cuda.is_available() else None)
            }
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            print(f"✅ Model info saved: {info_path}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False

    def load(self, path="models/model.pt"):
        """Load model with proper error handling"""
        try:
            if not os.path.exists(path):
                print(f"⚠️  Model file not found: {path}")
                return False
            
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict)
            file_size_mb = os.path.getsize(path) / (1024*1024)
            print(f"✅ Model loaded: {path} ({file_size_mb:.2f} MB)")
            
            # Load model info if exists
            info_path = path.replace('.pt', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                print(f"📊 Model info: {model_info.get('parameter_stats', {})}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False 