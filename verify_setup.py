
import torch
from model import build_llama
from config import *
from datasets import load_dataset
from dataset import StreamingLanguageModelDataset
from torch.utils.data import DataLoader

def verify():
    print("Verifying Model Configuration...")
    vocab_size = 100277
    model = build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    # Check if roughly 7M (Expect ~6.4M embeddings + ~0.5M layers -> ~7M)
    if n_params < 5_000_000 or n_params > 10_000_000:
        print("WARNING: Model size seems off from 7M target.")
    else:
        print("Model size looks correct (~7M).")

    print("\nVerifying Dataset Loading (Cosmopedia - web_samples_v2)...")
    try:
        ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
        dl = DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, "cl100k_base"), batch_size=1)
        batch = next(iter(dl))
        print("Successfully loaded one batch from Cosmopedia.")
        print(f"Input shape: {batch['input_ids'].shape}")
    except Exception as e:
        print(f"FAILED to load dataset: {e}")

if __name__ == "__main__":
    verify()
