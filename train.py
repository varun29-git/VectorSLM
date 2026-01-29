import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from collections import deque
from datasets import load_dataset
import random
import math
import logging
from typing import Callable, Optional, Dict

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset

# Configure Logging
Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{MODEL_FOLDER}/training.log")
    ]
)

# Suppress noise from third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

class ChinchillaScheduler:
    """
    Custom Scheduler for Chinchilla-Optimal Training Pipeline.
    
    Schedule:
    - Phase 1 (0 -> PHASE1_DURATION):
        - 0 -> 180M: Constant LR_MAX
        - 180M -> End: Cosine Decay to LR_MIN_PHASE1
    - Phase 2 (PHASE1_DURATION -> End):
        - Start -> +50M: Linear Warmup to LR_MAX_PHASE2
        - +50M -> End: Cosine Decay to LR_MIN_PHASE2
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        
    def step(self, tokens_seen: int) -> float:
        lr = self._calculate_lr(tokens_seen)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _calculate_lr(self, tokens_seen: int) -> float:
        # PHASE 1
        if tokens_seen < PHASE1_DURATION:
            if tokens_seen < 180_000_000:
                return LR_MAX
            
            # Cosine Decay
            progress = (tokens_seen - 180_000_000) / (PHASE1_DURATION - 180_000_000)
            progress = max(0.0, min(1.0, progress))
            return self._cosine_decay(LR_MAX, LR_MIN_PHASE1, progress)

        # PHASE 2
        else:
            phase2_tokens = tokens_seen - PHASE1_DURATION
            
            # Linear Warmup (50M tokens)
            if phase2_tokens < 50_000_000:
                progress = phase2_tokens / 50_000_000
                # Linear Interpolation
                return LR_MIN_PHASE1 + (LR_MAX_PHASE2 - LR_MIN_PHASE1) * progress
                
            # Cosine Decay (Remaining)
            else:
                decay_tokens = phase2_tokens - 50_000_000
                total_decay_duration = PHASE2_DURATION - 50_000_000
                progress = decay_tokens / total_decay_duration
                progress = max(0.0, min(1.0, progress))
                return self._cosine_decay(LR_MAX_PHASE2, LR_MIN_PHASE2, progress)

    def _cosine_decay(self, start_val: float, end_val: float, progress: float) -> float:
        cosine_coeff = 0.5 * (1 + math.cos(math.pi * progress))
        return end_val + (start_val - end_val) * cosine_coeff


def get_model(vocab_size: int) -> nn.Module:
    return build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )

def validate(model: nn.Module, dataset_name: str, device: torch.device, steps: int = 50) -> float:
    """Runs a quick validation loop."""
    logging.info(f"Running Validation for {dataset_name}...")
    model.eval()
    
    try:
        if dataset_name == "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English":
            ds = load_dataset(dataset_name, split="validation", streaming=True).rename_column("context", "text")
        elif dataset_name == "HuggingFaceFW/fineweb-edu":
            ds = load_dataset(dataset_name, name="sample-10BT", split="validation", streaming=True) # Fallback to train if no val? usually FW has distinct splits or we just use streaming train
            # FineWeb sample-10BT often doesn't split in streaming easily without config. assuming 'train' for now if fails.
        else:
            ds = load_dataset(dataset_name, split="validation", streaming=True)
    except Exception:
        logging.warning(f"Could not load validation split for {dataset_name}. Using 'train' stream for validation proxy.")
        try:
             ds = load_dataset(dataset_name, split="train", streaming=True)
             if dataset_name == "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English":
                 ds = ds.rename_column("context", "text")
        except:
             logging.error("Failed to load dataset for validation.")
             return 0.0

    val_dataset = StreamingLanguageModelDataset(ds, SEQ_LEN, "cl100k_base")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    val_loss_accum = 0.0
    steps_done = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            for i, batch in enumerate(val_loader):
                if i >= steps: break
                
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss_accum += loss.item()
                steps_done += 1
    
    avg_loss = val_loss_accum / max(steps_done, 1)
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss


def run_training_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: ChinchillaScheduler,
    primary_ds_name: str,
    mixing_strategy: Callable[[float], float], 
    duration: int,
    phase_name: str,
    global_tracker: Dict
):
    """
    Unified Training Loop for Mixed Phases.
    
    Args:
        mixing_strategy: Function(progress) -> tiny_stories_ratio
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    logging.info("="*60)
    logging.info(f"STARTING PHASE: {phase_name}")
    logging.info(f"Duration: {duration:,} tokens")
    logging.info("="*60)
    
    # Initialize Datasets
    ds_ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    if primary_ds_name == "incredible45/Gutenberg-BookCorpus-Cleaned-Data-English":
        ds_primary = load_dataset(primary_ds_name, split="train", streaming=True).rename_column("context", "text")
    elif primary_ds_name == "HuggingFaceFW/fineweb-edu":
        ds_primary = load_dataset(primary_ds_name, name="sample-10BT", split="train", streaming=True)
    else:
        ds_primary = load_dataset(primary_ds_name, split="train", streaming=True)

    dl_primary = DataLoader(StreamingLanguageModelDataset(ds_primary, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)
    dl_ts = DataLoader(StreamingLanguageModelDataset(ds_ts, SEQ_LEN, "cl100k_base"), batch_size=BATCH_SIZE, num_workers=0)
    
    iter_primary = iter(dl_primary)
    iter_ts = iter(dl_ts)
    
    total_phase_tokens = 0
    pbar = tqdm(total=duration // (BATCH_SIZE * SEQ_LEN), desc="Training", dynamic_ncols=True)
    
    loss_window = deque(maxlen=50)
    step = 0
    
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    while total_phase_tokens < duration:
        step += 1
        
        # 1. Update Learning Rate
        current_lr = scheduler.step(global_tracker['tokens_seen'])
        
        # 2. Determine Batch Source
        progress = total_phase_tokens / duration
        progress = max(0.0, min(1.0, progress))
        ts_ratio = mixing_strategy(progress)
        
        use_ts = random.random() < ts_ratio
        
        # 3. Fetch Batch
        try:
            batch = next(iter_ts) if use_ts else next(iter_primary)
        except StopIteration:
            # Restart exhausted iterator
            if use_ts:
                iter_ts = iter(dl_ts)
                batch = next(iter_ts)
            else:
                iter_primary = iter(dl_primary)
                batch = next(iter_primary)
                
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        batch_tokens = input_ids.numel()
        
        total_phase_tokens += batch_tokens
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(1)
        
        # 4. Forward & Backward
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        
        # 5. Gradient Accumulation Step
        if step % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        # 6. Logging
        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)
        
        eta_seconds = (ESTIMATED_TOTAL_TOKENS - global_tracker['tokens_seen']) / max(global_tracker['tokens_seen'] / (time.time() - global_tracker['start_time'] + 1e-6), 1e-6)
        eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"
        
        pbar.set_description(f"TS% {ts_ratio*100:.1f} | LR {current_lr:.2e} | ETA {eta_str} | L {avg_loss:.4f}")
        
    pbar.close()
    logging.info(f"Phase Complete. Tokens Processed: {total_phase_tokens:,}")
    validate(model, primary_ds_name, device)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    vocab_size = 100277 # cl100k_base size
    model = get_model(vocab_size).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR_MAX, # Initial, but scheduler overrides
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    scheduler = ChinchillaScheduler(optimizer)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Parameters: {n_params:,}")
    logging.info(f"Total Target Tokens: {ESTIMATED_TOTAL_TOKENS:,}")
    
    global_tracker = {
        'start_time': time.time(),
        'tokens_seen': 0
    }

    # ============================================================
    # PHASE 1: Gutenberg + Decaying TinyStories
    # ============================================================
    # Strategy: Linear Decay 0.6 -> 0.1
    def phase1_strategy(progress):
        return PHASE1_TS_START - (PHASE1_TS_START - PHASE1_TS_END) * progress

    run_training_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        primary_ds_name="incredible45/Gutenberg-BookCorpus-Cleaned-Data-English",
        mixing_strategy=phase1_strategy,
        duration=PHASE1_DURATION,
        phase_name="Phase 1: Gutenberg + TS (Decay)",
        global_tracker=global_tracker
    )
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/checkpoint_phase1.pt")

    # ============================================================
    # PHASE 2: FineWeb + Fixed TinyStories
    # ============================================================
    # Strategy: Fixed 0.1
    def phase2_strategy(progress):
        return PHASE2_TS_FIXED

    run_training_phase(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        primary_ds_name="HuggingFaceFW/fineweb-edu",
        mixing_strategy=phase2_strategy,
        duration=PHASE2_DURATION,
        phase_name="Phase 2: FineWeb + TS (Fixed)",
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/model_final.pt")
    
    total_time = time.time() - global_tracker['start_time']
    logging.info("="*60)
    logging.info(f"TRAINING COMPLETE. Total Time: {total_time/3600:.2f} hours")
    logging.info(f"Total Tokens Processed: {global_tracker['tokens_seen']:,}")
    logging.info("="*60)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
