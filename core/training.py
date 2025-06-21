# training.py
import torch
from torch.utils.data import IterableDataset
import os
import random
import json
from core.ai_tools import OpenAIKnowledgeExtractor
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torch.nn.utils.rnn import pad_sequence

# Setup logging for live demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log', mode='w')
    ]
)

# --- Streaming Dataloader ---
class StreamingDataset(IterableDataset):
    def __init__(self, data_dirs, shard_id=0, num_shards=1, augment_fn=None, auto_fetch_if_empty=True):
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.augment_fn = augment_fn
        self.files = self._discover_files()
        if auto_fetch_if_empty and not self.files:
            print("[INFO] No data found. Bootstrapping with OpenAI knowledge extraction...")
            def get_master_topics():
                path = 'datasets/topic_status.json'
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return list(json.load(f).keys())
                # fallback to minimal list if file does not exist
                return ["code", "general knowledge", "conversation"]
            topics = get_master_topics()
            extractor = OpenAIKnowledgeExtractor()
            extractor.extract(topics, max_cost=2.80, batch_size=4)
            self.files = self._discover_files()

    def _discover_files(self):
        files = []
        for d in self.data_dirs:
            for root, _, fs in os.walk(d):
                for f in fs:
                    if f.endswith('.jsonl') or f.endswith('.txt'):
                        files.append(os.path.join(root, f))
        files = sorted(files)
        return files[self.shard_id::self.num_shards]

    def __iter__(self):
        for file in self.files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            x, y = self._parse(data)
                            if self.augment_fn:
                                x, y = self.augment_fn(x, y)
                            yield x, y
                        except Exception:
                            continue
            except Exception as e:
                print(f"[WARNING] Skipping file {file}: {e}")

    def _parse(self, data):
        x = torch.tensor([ord(c) for c in str(data.get('input', ''))], dtype=torch.long)
        y = torch.tensor([ord(c) for c in str(data.get('output', ''))], dtype=torch.long)
        return x, y

def auto_ingest(source, dest_dir):
    pass

def pad_collate(batch):
    xs, ys = zip(*batch)
    # Find the max length in both xs and ys
    max_len = max(max(x.size(0) for x in xs), max(y.size(0) for y in ys))
    xs_padded = pad_sequence([torch.cat([x, torch.zeros(max_len - x.size(0), dtype=x.dtype)]) if x.size(0) < max_len else x for x in xs], batch_first=True, padding_value=0)
    ys_padded = pad_sequence([torch.cat([y, torch.zeros(max_len - y.size(0), dtype=y.dtype)]) if y.size(0) < max_len else y for y in ys], batch_first=True, padding_value=0)
    batch_size = xs_padded.size(0)
    audio = torch.zeros(batch_size, 1, 128, 128)
    video = torch.zeros(batch_size, 3, 16, 64, 64)
    return xs_padded, audio, video, ys_padded

# --- Single-node Training ---
def train_model(model, dataset, epochs=5, save_path="models/model.pt"):
    model.train()
    
    # Better optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=pad_collate)  # Smaller batch size
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (x, audio, video, y) in enumerate(loader):
            try:
                optimizer.zero_grad()
                output = model(x, audio, video, y)
                logits = output['text_feat'] if isinstance(output, dict) and 'text_feat' in output else output
                
                # Log shapes for debugging
                logging.info(f"Batch {batch_idx}: logits shape {logits.shape}, y shape {y.shape}")
                
                # Fix tensor shapes for loss calculation
                if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = y.view(-1)
                    # Ensure targets are within vocab range
                    vocab_size = logits.size(-1)
                    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
                else:
                    logits_flat = logits
                    targets_flat = y
                
                # Validate shapes before loss calculation
                if logits_flat.size(0) != targets_flat.size(0):
                    logging.warning(f"Shape mismatch: logits {logits_flat.shape}, targets {targets_flat.shape}")
                    continue
                
                loss = criterion(logits_flat, targets_flat)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                logging.info(f"Batch {batch_idx}: Loss {loss.item():.4f}")
                
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            logging.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                model.save(save_path + ".best")
                logging.info(f"New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logging.info(f"Early stopping after {epoch+1} epochs")
                    break
        else:
            logging.warning(f"Epoch {epoch+1}/{epochs}: No valid batches processed")
    
    # Save model with proper error handling
    if model.save(save_path):
        logging.info(f"Model saved successfully to {save_path}")
    else:
        logging.error(f"Failed to save model to {save_path}")
    
    return best_loss

# --- Distributed Training ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, model_class, dataset_fn, epochs=5, *model_kwargs):
    setup(rank, world_size)
    torch.manual_seed(42)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if model_kwargs and isinstance(model_kwargs[0], dict):
        model_kwargs = model_kwargs[0]
    else:
        model_kwargs = {}
    model = model_class(**model_kwargs).to(device)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    dataset = dataset_fn(rank, world_size)
    if isinstance(dataset, IterableDataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=pad_collate)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(epochs):
        ddp_model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (x, audio, video, y) in enumerate(loader):
            try:
                x, y = x.to(device), y.to(device)
                audio = audio.to(device) if audio is not None else torch.zeros(x.size(0), 1, 128, 128).to(device)
                video = video.to(device) if video is not None else torch.zeros(x.size(0), 3, 16, 64, 64).to(device)
                
                optimizer.zero_grad()
                output = ddp_model(x, audio, video, y)
                logits = output['text_feat'] if isinstance(output, dict) and 'text_feat' in output else output
                
                # Fix tensor shapes for loss calculation
                if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = y.view(-1)
                    # Ensure targets are within vocab range
                    vocab_size = logits.size(-1)
                    targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
                else:
                    logits_flat = logits
                    targets_flat = y
                
                # Validate shapes before loss calculation
                if logits_flat.size(0) != targets_flat.size(0):
                    print(f"Rank {rank}: Shape mismatch: logits {logits_flat.shape}, targets {targets_flat.shape}")
                    continue
                
                loss = criterion(logits_flat, targets_flat)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Rank {rank}: Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Rank {rank}, Epoch {epoch+1}/{epochs}: No valid batches processed")
    
    # Save model
    if rank == 0:  # Only save from rank 0
        torch.save(model.state_dict(), "models/model.pt")
        print(f"✅ Model saved from rank {rank}")
    
    cleanup()

def run_distributed_training(model_class, dataset_fn, world_size=2, epochs=5, **model_kwargs):
    mp.spawn(train_ddp, args=(world_size, model_class, dataset_fn, epochs, model_kwargs), nprocs=world_size, join=True)

def elastic_scale_hook():
    pass

# --- Async/Batch Training Stubs ---
def batch_train():
    pass

def async_train():
    pass

def train_on_topic(model, topic, dataset_dir='datasets', save_path='models/model.pt'):
    """
    Interactive topic-wise training with real-time monitoring and detailed logging.
    """
    import os
    import json
    import time
    from datetime import datetime
    from core.ai_tools import OpenAIKnowledgeExtractor
    import logging

    print(f"\n{'='*100}")
    print(f"🎯 TRAINING TOPIC: {topic.upper()}")
    print(f"⏰ STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")

    # Check if topic already trained
    trained_topics_file = "logs/trained_topics.txt"
    trained_topics = set()
    if os.path.exists(trained_topics_file):
        with open(trained_topics_file, 'r') as f:
            trained_topics = set(line.strip() for line in f if line.strip())
    
    if topic in trained_topics:
        print(f"⚠️  Topic '{topic}' already trained. Skipping...")
        return None

    # 1. Show current model stats with accurate parameter counting
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = param_count * 4 / (1024*1024)  # 4 bytes per parameter (float32)
    
    print(f"📊 CURRENT MODEL STATS:")
    print(f"   Parameters: {param_count:,}")
    print(f"   Expected Model Size: {model_size_mb:.2f} MB")
    print(f"   Vocab Size: 250,000")
    if os.path.exists(save_path):
        actual_size = os.path.getsize(save_path) / (1024*1024)
        print(f"   Actual Model Size: {actual_size:.2f} MB")
        if abs(actual_size - model_size_mb) > 10:  # More than 10MB difference
            print(f"   ⚠️  WARNING: Size mismatch! Expected {model_size_mb:.2f}MB, got {actual_size:.2f}MB")

    # 2. Extract data for the topic
    print(f"\n🔍 EXTRACTING DATA FOR TOPIC: '{topic}'")
    extractor = OpenAIKnowledgeExtractor()
    start_time = time.time()
    extractor.extract([topic], max_cost=2.0, batch_size=4)
    extraction_time = time.time() - start_time
    print(f"✅ Data extraction completed in {extraction_time:.2f}s")

    # 3. Find and analyze the dataset
    topic_file = None
    for file in os.listdir(dataset_dir):
        if topic.replace(' ', '_').lower() in file.lower() and file.endswith('.jsonl'):
            topic_file = os.path.join(dataset_dir, file)
            break
    if not topic_file:
        topic_file = os.path.join(dataset_dir, 'knowledge.jsonl')  # fallback

    # Count dataset size
    dataset_size = 0
    dataset_lines = 0
    try:
        with open(topic_file, 'r', encoding='utf-8') as f:
            for line in f:
                dataset_lines += 1
                try:
                    data = json.loads(line)
                    if 'input' in data and 'output' in data:
                        dataset_size += len(data['input']) + len(data['output'])
                except:
                    continue
    except Exception as e:
        print(f"⚠️  Error reading dataset: {e}")

    print(f"📚 DATASET STATS FOR TOPIC '{topic}':")
    print(f"   File: {os.path.basename(topic_file)}")
    print(f"   Lines: {dataset_lines:,}")
    print(f"   Characters: {dataset_size:,}")
    print(f"   File Size: {os.path.getsize(topic_file) / (1024*1024):.2f} MB")

    # 4. Train on this topic's data with better loss monitoring
    print(f"\n🚀 TRAINING ON TOPIC: '{topic}'")
    dataset = StreamingDataset(data_dirs=[topic_file])
    
    # Enhanced training with progress tracking
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=pad_collate)  # Smaller batch size
    
    total_batches = len(loader)
    epoch_loss = 0
    batch_count = 0
    
    print(f"   🎯 Training Configuration:")
    print(f"      Topic: {topic}")
    print(f"      Epochs: 1")
    print(f"      Batch Size: 8")
    print(f"      Total Batches: {total_batches}")
    print(f"      Learning Rate: 1e-4")
    print(f"      Dataset: {os.path.basename(topic_file)}")
    
    start_time = time.time()
    
    print(f"\n   📈 TRAINING PROGRESS:")
    for batch_idx, (x, audio, video, y) in enumerate(loader):
        batch_count += 1
        try:
            optimizer.zero_grad()
            
            output = model(x, audio, video, y)
            logits = output['text_feat'] if isinstance(output, dict) and 'text_feat' in output else output
            
            # Fix tensor shapes for loss calculation
            if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                # Ensure targets are within vocab range
                vocab_size = logits.size(-1)
                targets_flat = torch.clamp(targets_flat, 0, vocab_size - 1)
            else:
                logits_flat = logits
                targets_flat = y
                
            # Validate shapes before loss calculation
            if logits_flat.size(0) != targets_flat.size(0):
                print(f"      ⚠️  Shape mismatch in batch {batch_idx}: logits {logits_flat.shape}, targets {targets_flat.shape}")
                continue
            
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Real-time progress with loss monitoring
            if batch_idx % 1 == 0 or batch_idx == total_batches - 1:  # Show every batch
                avg_loss = epoch_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / total_batches * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx < total_batches - 1 else 0
                
                print(f"      [{batch_idx+1:3d}/{total_batches}] Loss: {avg_loss:.4f} | Progress: {progress:5.1f}% | ETA: {eta:.1f}s | Topic: {topic}")
                
                # Warn if loss is too high
                if avg_loss > 8.0:
                    print(f"      ⚠️  WARNING: High loss ({avg_loss:.4f}). Consider adjusting learning rate or batch size.")
                    
        except Exception as e:
            print(f"      ❌ Error in batch {batch_idx}: {e}")
            continue

    training_time = time.time() - start_time
    final_loss = epoch_loss / batch_count

    # 5. Final stats and save
    print(f"\n✅ TRAINING COMPLETED FOR TOPIC: '{topic}'")
    print(f"   Final Loss: {final_loss:.4f}")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Total Time: {training_time + extraction_time:.2f}s")
    
    # Save model
    model.save(save_path)
    print(f"💾 Model saved to: {save_path}")
    
    # 6. Post-training stats
    new_param_count = sum(p.numel() for p in model.parameters())
    actual_model_size = os.path.getsize(save_path) / (1024*1024)
    expected_model_size = new_param_count * 4 / (1024*1024)
    
    print(f"\n📈 POST-TRAINING STATS FOR TOPIC '{topic}':")
    print(f"   Model Parameters: {new_param_count:,}")
    print(f"   Expected Model Size: {expected_model_size:.2f} MB")
    print(f"   Actual Model Size: {actual_model_size:.2f} MB")
    print(f"   Vocab Size: 250,000")
    
    # Mark topic as trained
    os.makedirs('logs', exist_ok=True)
    with open(trained_topics_file, 'a') as f:
        f.write(f"{topic}\n")
    print(f"✅ Topic '{topic}' marked as trained")
    
    # Log to file
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic,
        "parameters": new_param_count,
        "expected_model_size_mb": expected_model_size,
        "actual_model_size_mb": actual_model_size,
        "dataset_lines": dataset_lines,
        "dataset_chars": dataset_size,
        "final_loss": final_loss,
        "training_time": training_time,
        "extraction_time": extraction_time
    }
    
    with open("logs/topic_training_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"📝 Logged to: logs/topic_training_log.jsonl")
    print(f"{'='*100}\n")
    
    return log_entry 