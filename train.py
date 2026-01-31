import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from collections import deque
from datasets import load_dataset
from tokenizers import Tokenizer

from config import *
from model import build_llama
from dataset import StreamingLanguageModelDataset
import random
import math

TOTAL_TRAINING_TOKENS = 1_000_000_000

def get_lr(tokens_seen):
    # Simple Cosine Decay for 100M tokens
    progress = tokens_seen / TOTAL_TRAINING_TOKENS
    progress = max(0.0, min(1.0, progress))
    
    target_min = 1e-5
    target_max = LR
    
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return target_min + (target_max - target_min) * cosine_decay

def get_model(vocab_size):
    return build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )

def validate(model, dataset_name, device, steps=50):
    """Runs a quick validation loop on the validation or test split."""
    print(f"\n--- Running Validation for {dataset_name} ---")
    model.eval()
    
    ds = None
    try:
        # Try validation split first
        ds = load_dataset(dataset_name, split="validation", streaming=True)
    except Exception:
        try:
            # Fallback to test split
            ds = load_dataset(dataset_name, split="test", streaming=True)
        except Exception:
            print(f"No validation/test split found for {dataset_name}. Skipping validation.")
            model.train()
            return None

    val_dataset = StreamingLanguageModelDataset(
        ds,
        seq_len=SEQ_LEN,
        tokenizer_name="cl100k_base"
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    val_loss_accum = 0.0
    steps_done = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            for i, batch in enumerate(val_loader):
                if i >= steps:
                    break
                
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                
                logits = model(input_ids)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                val_loss_accum += loss.item()
                steps_done += 1
    
    avg_val_loss = val_loss_accum / max(steps_done, 1)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    model.train()
    return avg_val_loss


def train_sequential_phases(model, optimizer, scaler, vocab_size, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load tokenizer.json ({e})")
        return

    # Phase Configs
    # Total: 1B
    
    phases = [
        {"name": "Cosmopedia (Knowledge)", "tokens": 550_000_000, "dataset_id": "HuggingFaceTB/cosmopedia", "subset": "web_samples_v2", "map_fn": None},
        {"name": "FineWeb-Edu (Academic)", "tokens": 300_000_000, "dataset_id": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "map_fn": None},
        {"name": "Evol-Instruct (Code)", "tokens": 150_000_000, "dataset_id": "nickrosh/Evol-Instruct-Code-80k-v1", "subset": None, "map_fn": lambda x: {"text": f"{x['instruction']}\n{x['output']}"}}
    ]

    pbar = tqdm(total=TOTAL_TRAINING_TOKENS // (BATCH_SIZE * SEQ_LEN), dynamic_ncols=True)
    loss_window = deque(maxlen=50)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    
    model.train()

    current_phase_idx = 0
    phase_tokens_processed = 0
    
    # We iterate through phases sequentially
    for i, phase in enumerate(phases):
        print("\n" + "=" * 80)
        print(f"STARTING PHASE {i+1}: {phase['name']}")
        print(f"Goal: {phase['tokens']:,} tokens")
        print("=" * 80)
        
        # Load Dataset for this phase
        print(f"Loading {phase['dataset_id']}...")
        if phase['subset']:
            ds_raw = load_dataset(phase['dataset_id'], phase['subset'], split="train", streaming=True)
        else:
            ds_raw = load_dataset(phase['dataset_id'], split="train", streaming=True)
            
        if phase['map_fn']:
            ds = ds_raw.map(phase['map_fn'], remove_columns=["instruction", "output"])
        else:
            ds = ds_raw
            
        dl = DataLoader(
            StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), 
            batch_size=BATCH_SIZE, 
            num_workers=4, 
            pin_memory=True
        )
        iterator = iter(dl)
        
        target_tokens_for_phase = phase['tokens']
        tokens_in_this_phase = 0
        
        while tokens_in_this_phase < target_tokens_for_phase:
            step += 1
            
            # LR Schedule (Global)
            current_lr = get_lr(global_tracker['tokens_seen'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dl)
                batch = next(iterator)
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            batch_tokens = input_ids.numel()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward() # type: ignore

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Updates
            global_tracker['tokens_seen'] += batch_tokens
            tokens_in_this_phase += batch_tokens
            pbar.update(1)

            loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
            avg_loss = sum(loss_window) / len(loss_window)
            
            eta_str = "??"
            elapsed = time.time() - global_tracker['start_time']
            rate = global_tracker['tokens_seen'] / max(elapsed, 1e-6)
            remaining = TOTAL_TRAINING_TOKENS - global_tracker['tokens_seen']
            eta_seconds = remaining / max(rate, 1e-6)
            eta_str = f"{int(eta_seconds//3600)}h {int((eta_seconds%3600)//60)}m"

            pbar.set_postfix({
                "Ph": i+1,
                "LR": f"{current_lr:.1e}",
                "L": f"{avg_loss:.2f}",
                "ETA": eta_str
            })

    pbar.close()
    print(f"Training Complete. Total Tokens: {global_tracker['tokens_seen']:,}")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For Mac MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    vocab_size = VOCAB_SIZE
    model = get_model(vocab_size).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Initialize Optimizer
    optimizer = None
    try:
        import bitsandbytes as bnb
        print("Using 8-bit AdamW optimizer via bitsandbytes...")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.99),
            eps=1e-8
        )
    except Exception as e:
        print(f"Warning: bitsandbytes failed to load ({e}). Fallback to standard AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=0.1,
            betas=(0.9, 0.99),
            eps=1e-8
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    # Global Progress Tracker
    global_tracker = {
        'start_time': time.time(),
        'tokens_seen': 0
    }

    train_sequential_phases(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        vocab_size=vocab_size,
        global_tracker=global_tracker
    )
    
    torch.save(model.state_dict(), f"{MODEL_FOLDER}/model_final.pt")
    
    total_time = time.time() - global_tracker['start_time']
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Total Time: {total_time/3600:.2f} hours")
    print(f"Total Tokens Processed: {global_tracker['tokens_seen']:,}")
    print("=" * 80)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
