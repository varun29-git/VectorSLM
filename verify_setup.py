
import torch
from model import build_llama
from config import *
from datasets import load_dataset
from dataset import StreamingLanguageModelDataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

def verify():
    print("Verifying Model Configuration...")
    # Expect VOCAB_SIZE = 4096
    vocab_size = VOCAB_SIZE 
    print(f"Vocab Size: {vocab_size}")
    
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
    
    # Expected: ~10.1M
    # Embedding: 16384 * 256 = 4,194,304
    # Layers: ~6M
    # Total roughly ~10.1M

    
    print("\nVerifying Tokenizer & Dataset Loading...")
    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
        print("Loaded tokenizer.json")
        
        # 1. Cosmopedia
        print("Checking Cosmopedia...")
        ds = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
        dl = DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), batch_size=1)
        next(iter(dl))
        print("OK: Cosmopedia")

        # 2. FineWeb-Edu
        print("Checking FineWeb-Edu...")
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        dl = DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), batch_size=1)
        next(iter(dl))
        print("OK: FineWeb-Edu")

        # 3. Evol-Instruct
        print("Checking Evol-Instruct...")
        ds_raw = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        def format_code(example):
            return {"text": f"{example['instruction']}\n{example['output']}"}
        ds = ds_raw.map(format_code, remove_columns=["instruction", "output"])
        
        dl = DataLoader(StreamingLanguageModelDataset(ds, SEQ_LEN, tokenizer), batch_size=1)
        next(iter(dl))
        print("OK: Evol-Instruct")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    verify()

if __name__ == "__main__":
    verify()
