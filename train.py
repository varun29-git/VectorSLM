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

TOTAL_TRAINING_TOKENS = 3_600_000_000


        # Updates
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(1)

        loss_window.append(loss.item() * GRAD_ACCUM_STEPS)
        avg_loss = sum(loss_window) / len(loss_window)

        pbar.set_postfix({
            "LR": f"{current_lr:.1e}",
            "L": f"{avg_loss:.2f}",
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

    train_mixed_strategy(
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