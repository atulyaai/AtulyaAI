# system_check.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import hashlib
import pickle
import gzip
from datetime import datetime
from collections import defaultdict
import numpy as np

class SystemChecker:
    def __init__(self):
        self.issues = []
        self.optimizations = []
        self.duplicates = []
        self.redundancies = []
        
    def check_model_files(self):
        """Check all model files for issues"""
        print("🔍 Checking model files...")
        
        model_files = [
            'models/model.pt',
            'models/tokenizer.json',
            'models/tokenizer.py',
            'models/modalities.py',
            'models/model.py'
        ]
        
        for file_path in model_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024*1024)
                print(f"✅ {file_path}: {file_size:.2f} MB")
                
                # Check for corruption
                try:
                    if file_path.endswith('.pt'):
                        state = torch.load(file_path, map_location='cpu')
                        param_count = sum(p.numel() for p in state.values())
                        print(f"   Parameters: {param_count:,}")
                    elif file_path.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        print(f"   JSON keys: {len(data)}")
                except Exception as e:
                    self.issues.append(f"Corrupted file: {file_path} - {e}")
            else:
                self.issues.append(f"Missing file: {file_path}")
    
    def check_duplicates(self):
        """Find duplicate files and parameters"""
        print("🔍 Checking for duplicates...")
        
        # Check for duplicate parameter names
        try:
            from models.model import MultimodalModel
            model = MultimodalModel()
            
            param_names = [name for name, _ in model.named_parameters()]
            duplicates = [name for name in set(param_names) if param_names.count(name) > 1]
            
            if duplicates:
                self.duplicates.extend(duplicates)
                print(f"⚠️  Found {len(duplicates)} duplicate parameter names")
            else:
                print("✅ No duplicate parameter names found")
                
        except Exception as e:
            self.issues.append(f"Error checking duplicates: {e}")
    
    def check_redundancies(self):
        """Find redundant parameters and layers"""
        print("🔍 Checking for redundancies...")
        
        try:
            from models.model import MultimodalModel
            model = MultimodalModel()
            
            redundant_layers = []
            for name, param in model.named_parameters():
                # Check for zero parameters
                if param.numel() > 0 and torch.all(param == 0):
                    redundant_layers.append(f"{name}: all zeros")
                
                # Check for very small parameters
                if param.numel() > 0 and torch.all(torch.abs(param) < 1e-6):
                    redundant_layers.append(f"{name}: very small values")
                
                # Check for constant parameters
                if param.numel() > 1 and torch.all(param == param[0]):
                    redundant_layers.append(f"{name}: constant values")
            
            if redundant_layers:
                self.redundancies.extend(redundant_layers)
                print(f"⚠️  Found {len(redundant_layers)} redundant layers")
            else:
                print("✅ No redundant layers found")
                
        except Exception as e:
            self.issues.append(f"Error checking redundancies: {e}")
    
    def optimize_model(self):
        """Optimize model for better performance"""
        print("⚡ Optimizing model...")
        
        try:
            from models.model import MultimodalModel
            model = MultimodalModel()
            
            if os.path.exists('models/model.pt'):
                model.load('models/model.pt')
            
            # 1. Remove redundant parameters
            for name, param in model.named_parameters():
                if param.numel() > 0 and torch.all(param == 0):
                    param.data = torch.randn_like(param) * 0.01
                    self.optimizations.append(f"Initialized zero parameters in {name}")
            
            # 2. Apply weight normalization
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    with torch.no_grad():
                        norm = torch.norm(param.data, dim=1, keepdim=True)
                        param.data = param.data / (norm + 1e-8)
                    self.optimizations.append(f"Applied weight normalization to {name}")
            
            # 3. Save optimized model
            model.save('models/model_optimized.pt')
            print("✅ Model optimized and saved")
            
        except Exception as e:
            self.issues.append(f"Error optimizing model: {e}")
    
    def check_vocab_size(self):
        """Check and fix vocab size issues"""
        print("🔤 Checking vocabulary size...")
        
        try:
            from models.tokenizer import HFTokenizer
            tokenizer = HFTokenizer(vocab_size=250000)
            
            if os.path.exists('models/tokenizer.json'):
                tokenizer.load('models/tokenizer.json')
                current_vocab = tokenizer.vocab_size
                print(f"Current vocab size: {current_vocab:,}")
                
                if current_vocab < 100000:
                    self.issues.append(f"Vocab size too small: {current_vocab:,} (should be 250K+)")
                    self.optimizations.append("Retrain tokenizer with larger vocab size")
                else:
                    print("✅ Vocab size is adequate")
            else:
                self.issues.append("Tokenizer file not found")
                
        except Exception as e:
            self.issues.append(f"Error checking vocab: {e}")
    
    def check_training_data(self):
        """Check training data quality"""
        print("📚 Checking training data...")
        
        dataset_files = []
        for root, dirs, files in os.walk('datasets'):
            for file in files:
                if file.endswith('.jsonl'):
                    dataset_files.append(os.path.join(root, file))
        
        if not dataset_files:
            self.issues.append("No training data found")
            return
        
        total_samples = 0
        for file_path in dataset_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    total_samples += len(lines)
                    print(f"✅ {file_path}: {len(lines):,} samples")
            except Exception as e:
                self.issues.append(f"Error reading {file_path}: {e}")
        
        print(f"📊 Total training samples: {total_samples:,}")
        
        if total_samples < 1000:
            self.issues.append(f"Insufficient training data: {total_samples:,} samples")
            self.optimizations.append("Generate more training data")
    
    def check_loss_optimization(self):
        """Check and suggest loss optimization strategies"""
        print("📉 Checking loss optimization...")
        
        optimizations = [
            "Use learning rate scheduling (ReduceLROnPlateau)",
            "Apply gradient clipping (max_norm=1.0)",
            "Use label smoothing (0.1)",
            "Implement early stopping",
            "Use AdamW optimizer with weight decay",
            "Apply mixed precision training",
            "Use gradient checkpointing",
            "Implement curriculum learning",
            "Apply data augmentation",
            "Use ensemble methods"
        ]
        
        self.optimizations.extend(optimizations)
        print(f"✅ Added {len(optimizations)} loss optimization strategies")
    
    def generate_report(self):
        """Generate comprehensive system report"""
        print("\n" + "="*60)
        print("📊 SYSTEM CHECK REPORT")
        print("="*60)
        
        print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        
        print(f"\n🔧 OPTIMIZATIONS SUGGESTED ({len(self.optimizations)}):")
        for i, opt in enumerate(self.optimizations, 1):
            print(f"  {i}. {opt}")
        
        print(f"\n🔄 DUPLICATES FOUND ({len(self.duplicates)}):")
        for i, dup in enumerate(self.duplicates, 1):
            print(f"  {i}. {dup}")
        
        print(f"\n🗑️  REDUNDANCIES FOUND ({len(self.redundancies)}):")
        for i, red in enumerate(self.redundancies, 1):
            print(f"  {i}. {red}")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'issues': self.issues,
            'optimizations': self.optimizations,
            'duplicates': self.duplicates,
            'redundancies': self.redundancies
        }
        
        with open('logs/system_check_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: logs/system_check_report.json")
        print("="*60)

def main():
    """Run comprehensive system check"""
    print("🚀 Starting comprehensive system check...")
    
    checker = SystemChecker()
    
    # Run all checks
    checker.check_model_files()
    checker.check_duplicates()
    checker.check_redundancies()
    checker.optimize_model()
    checker.check_vocab_size()
    checker.check_training_data()
    checker.check_loss_optimization()
    
    # Generate report
    checker.generate_report()
    
    print("\n🎉 System check completed!")

if __name__ == '__main__':
    main() 