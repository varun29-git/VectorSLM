import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import warnings
import time
from contextlib import nullcontext
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
import random

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    BNB_IMPORT_ERROR = None
except Exception as exc:
    bnb = None
    BNB_AVAILABLE = False
    BNB_IMPORT_ERROR = exc

from config import *
from model import build_llama

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
FT_LR = 3e-5
FT_EPOCHS = 1 
FT_BATCH_SIZE = BATCH_SIZE
FT_SEQ_LEN = 1024
CHECKPOINT_PATH = f"{MODEL_FOLDER}/model_final.pt"

# --------------------------------------------------------------------------------
# Mappers
# --------------------------------------------------------------------------------
def map_smoltalk(x):
    # Keys: ['messages', ...]
    try:
        msgs = x['messages']
        text = ""
        for m in msgs:
            role = "User" if m['role'] == 'user' else "Assistant"
            text += f"{role}: {m['content']}\n\n"
        return {"text": text.strip()}
    except:
        return {"text": ""}

def map_tulu_code(x):
    # Keys: ['messages', ...]
    try:
        msgs = x['messages']
        text = ""
        for m in msgs:
            role = "User" if m['role'] == 'user' else "Assistant"
            text += f"{role}: {m['content']}\n\n"
        return {"text": text.strip()}
    except:
        return {"text": ""}

def map_slimorca(x):
    # Keys: ['conversations'] -> [{'from': 'system', 'value':...}, {'from': 'human'...}]
    convs = x['conversations']
    text = ""
    for c in convs:
        role = c['from']
        val = c['value']
        # Map roles
        if role == 'system':
            text += f"System: {val}\n"
        elif role == 'human':
            text += f"User: {val}\n\n"
        elif role == 'gpt':
            text += f"Assistant: {val}\n\n"
    return {"text": text.strip()}

# --------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------
class MixedInstructionDataset(Dataset):
    def __init__(self, tokenizer, max_length=1024, max_steps=50000): # 200M / (4*1024) approx 48k steps
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_steps = max_steps
        
        print("Loading datasets for mixing...")
        
        # SmolTalk 
        print("   - HuggingFaceTB/smoltalk (General) (65%)")
        ds_smol = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        ds_smol = ds_smol.map(map_smoltalk)
        ds_smol = ds_smol.select_columns(["text"])
        
        # Tulu-3-Code (15%)
        print("   - allenai/tulu-3-sft-personas-code (15%)")
        ds_code = load_dataset("allenai/tulu-3-sft-personas-code", split="train", streaming=True)
        ds_code = ds_code.map(map_tulu_code)
        ds_code = ds_code.select_columns(["text"])
        
        # SlimOrca 
        print("   - Open-Orca/SlimOrca (20%)")
        ds_orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
        ds_orca = ds_orca.map(map_slimorca)
        ds_orca = ds_orca.select_columns(["text"])
        
        # Interleave [0.65, 0.15, 0.20]
        probs = [0.65, 0.15, 0.20]
        self.mixed = interleave_datasets([ds_smol, ds_code, ds_orca], probabilities=probs, stopping_strategy="all_exhausted")
        self.iterator = iter(self.mixed)
        
    def __len__(self):
        # Virtual length for tqdm
        return self.max_steps
        
    def __getitem__(self, idx):
        # We ignore idx for streaming, just get next
        try:
            item = next(self.iterator)
            text = item['text']
            
            # Argilla 700 token cap check (approximate or precise)
            # We must tokenize first to check precise length
            tokens = self.tokenizer.encode(text).ids
            
            # Tokenize formatting
            eos_id = self.tokenizer.token_to_id("<EOS>")
            if eos_id is None: eos_id = 3
            tokens.append(eos_id)
            
            # Truncate to Sequence Length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            # Pad
            pad_id = self.tokenizer.token_to_id("<PAD>")
            if pad_id is None: pad_id = 1
            
            padding = [pad_id] * (self.max_length - len(tokens))
            tokens = tokens + padding
            
            x = torch.tensor(tokens, dtype=torch.long)
            y = x.clone()
            if len(padding) > 0:
                y[-len(padding):] = -100
            
            return {
                "input_ids": x[:-1],
                "targets": y[1:]
            }
            
        except StopIteration:
            self.iterator = iter(self.mixed)
            return self.__getitem__(idx)

# --------------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------------
def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def build_optimizer(model, device):
    if device.type == "cuda" and BNB_AVAILABLE:
        print("Optimizer: bitsandbytes AdamW8bit")
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=FT_LR,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
        )

    if BNB_IMPORT_ERROR is not None:
        print(f"bitsandbytes unavailable, using torch.optim.AdamW ({BNB_IMPORT_ERROR})")
    else:
        print("Using torch.optim.AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=FT_LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )


def train_finetune():
    device = resolve_device()
    print(f"Using device: {device}")

    Path(FINETUNE_MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Build Model
    vocab_size = tokenizer.get_vocab_size() # explicit check
    print(f"Building model (Vocab: {vocab_size})...")
    model = build_llama(
        vocab_size=vocab_size, 
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    # Load Checkpoint
    if Path(CHECKPOINT_PATH).exists():
        print(f"Loading pretrained weights from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found! Starting from scratch.")

    # Data
    # 300M tokens / (16 batch * 1024 seq) ~= 18311 steps
    train_dataset = MixedInstructionDataset(tokenizer, max_length=FT_SEQ_LEN, max_steps=18311)
    train_loader = DataLoader(
        train_dataset,
        batch_size=FT_BATCH_SIZE,
        num_workers=1,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    optimizer = build_optimizer(model, device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Loop
    model.train()
    print(f"Starting Fine-tuning (Mixed Strategy)...")
    
    pbar = tqdm(train_loader, desc=f"Fine-tuning", dynamic_ncols=True)
    total_loss = 0
    steps = 0
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
        targets = batch["targets"].to(device, non_blocking=(device.type == "cuda"))
        
        optimizer.zero_grad(set_to_none=True)
        
        with get_autocast_context(device):
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        steps += 1
        
        pbar.set_postfix({"Loss": f"{loss_val:.4f}", "Avg": f"{total_loss/steps:.4f}"})
        
        if steps % 500 == 0:
            torch.save(model.state_dict(), f"{FINETUNE_MODEL_FOLDER}/model_ft_step_{steps}.pt")
            
    print("Fine-tuning Complete.")
    torch.save(model.state_dict(), f"{FINETUNE_MODEL_FOLDER}/model_finetuned_final.pt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_finetune()
