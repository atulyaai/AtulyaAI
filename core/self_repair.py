# self_repair.py
import torch
import torch.nn as nn
import os
import json
import logging
from datetime import datetime
from models.model import MultimodalModel
from core.training import train_model, StreamingDataset

class SelfRepairEngine:
    def __init__(self, model_path="models/model.pt", config_path="configs/config.json"):
        self.model_path = model_path
        self.config_path = config_path
        self.repair_log = "logs/self_repair.log"
        self.health_threshold = 0.85
        self.evolution_cycles = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [SELF_REPAIR] %(message)s',
            handlers=[
                logging.FileHandler(self.repair_log, mode='a'),
                logging.StreamHandler()
            ]
        )
    
    def check_model_health(self, model):
        """Check model health and performance metrics"""
        try:
            # Test forward pass
            test_input = torch.randint(0, 1000, (1, 10))
            test_audio = torch.zeros(1, 1, 128, 128)
            test_video = torch.zeros(1, 3, 16, 64, 64)
            
            with torch.no_grad():
                output = model(test_input, test_audio, test_video)
            
            # Check for NaN or inf values
            has_nan = torch.isnan(output['text_feat']).any().item()
            has_inf = torch.isinf(output['text_feat']).any().item()
            
            # Calculate basic metrics
            param_count = sum(p.numel() for p in model.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            health_score = 1.0
            if has_nan or has_inf:
                health_score *= 0.5
            if param_count == 0:
                health_score *= 0.3
            
            return {
                "health_score": health_score,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "param_count": param_count,
                "grad_norm": grad_norm.item() if grad_norm is not None else 0.0,
                "status": "healthy" if health_score > self.health_threshold else "needs_repair"
            }
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return {
                "health_score": 0.0,
                "error": str(e),
                "status": "critical"
            }
    
    def optimize_model(self, model):
        """Apply model optimizations"""
        try:
            optimizations = []
            
            # 1. Weight pruning for sparsity
            total_params = 0
            pruned_params = 0
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    total_params += param.numel()
                    # Prune 10% of smallest weights
                    threshold = torch.quantile(param.abs(), 0.1)
                    mask = param.abs() > threshold
                    pruned_params += (~mask).sum().item()
                    param.data *= mask.float()
            
            if pruned_params > 0:
                optimizations.append(f"pruned {pruned_params}/{total_params} parameters")
            
            # 2. Quantization (simplified)
            if hasattr(model, 'text_model') and model.text_model:
                for param in model.text_model.parameters():
                    param.data = torch.round(param.data * 1000) / 1000  # Simple quantization
            
            # 3. Memory optimization
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logging.info(f"Applied optimizations: {', '.join(optimizations)}")
            return optimizations
            
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return []
    
    def evolve_model(self, model):
        """Evolve model architecture if needed"""
        try:
            # Check if model needs evolution
            health = self.check_model_health(model)
            
            if health["health_score"] < self.health_threshold:
                logging.info("Model health below threshold, initiating evolution...")
                
                # 1. Increase model capacity
                if hasattr(model, 'text_model') and model.text_model:
                    # Add more layers or increase hidden size
                    if hasattr(model.text_model, 'grow'):
                        model.text_model.grow(add_experts=1, add_layers=1)
                        logging.info("Added expert and layer to text model")
                
                # 2. Retrain on recent data
                dataset = StreamingDataset(data_dirs=['datasets'])
                train_model(model, dataset, epochs=2, save_path=self.model_path)
                
                # 3. Update evolution counter
                self.evolution_cycles += 1
                
                logging.info(f"Evolution cycle {self.evolution_cycles} completed")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Evolution failed: {e}")
            return False
    
    def auto_repair(self):
        """Main self-repair routine"""
        try:
            logging.info("Starting self-repair cycle...")
            
            # Load model
            if not os.path.exists(self.model_path):
                logging.warning("Model not found, creating new one...")
                model = MultimodalModel(vocab_size=50000)
                model.save(self.model_path)
            else:
                model = MultimodalModel(vocab_size=50000)
                model.load(self.model_path)
            
            # Check health
            health = self.check_model_health(model)
            logging.info(f"Health score: {health['health_score']:.3f}")
            
            # Apply optimizations
            optimizations = self.optimize_model(model)
            
            # Evolve if needed
            evolved = self.evolve_model(model)
            
            # Save repaired model
            model.save(self.model_path)
            
            # Update repair log
            repair_info = {
                "timestamp": datetime.now().isoformat(),
                "health_score": health["health_score"],
                "optimizations": optimizations,
                "evolved": evolved,
                "evolution_cycles": self.evolution_cycles
            }
            
            with open("logs/repair_history.json", "a") as f:
                f.write(json.dumps(repair_info) + "\n")
            
            logging.info("Self-repair cycle completed successfully")
            return repair_info
            
        except Exception as e:
            logging.error(f"Self-repair failed: {e}")
            return {"error": str(e)}
    
    def get_status(self):
        """Get current self-repair status"""
        try:
            status = {
                "last_repair": None,
                "evolution_cycles": self.evolution_cycles,
                "health_threshold": self.health_threshold,
                "repair_log_exists": os.path.exists(self.repair_log)
            }
            
            # Get last repair info
            if os.path.exists("logs/repair_history.json"):
                with open("logs/repair_history.json", "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_repair = json.loads(lines[-1])
                        status["last_repair"] = last_repair
            
            return status
            
        except Exception as e:
            return {"error": str(e)}

# Global instance
self_repair_engine = SelfRepairEngine()

def run_self_repair():
    """Run self-repair (called from CLI or scheduler)"""
    return self_repair_engine.auto_repair()

def get_repair_status():
    """Get repair status (called from web UI)"""
    return self_repair_engine.get_status() 