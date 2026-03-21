import random
import math
import os
import time
import warnings
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
    BNB_IMPORT_ERROR = None
except Exception as exc:
    bnb = None
    BNB_AVAILABLE = False
    BNB_IMPORT_ERROR = exc

from config import *
from dataset import StreamingLanguageModelDataset
from model import build_llama
from muon import SingleDeviceMuon

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_adamw_lr(step, total_steps):
    """Cosine LR schedule with linear warmup for the AdamW optimizer."""
    if step < WARMUP_STEPS:
        return LR * (step / WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = max(0.0, min(1.0, progress))
    min_lr = 1e-5
    return min_lr + 0.5 * (LR - min_lr) * (1 + math.cos(math.pi * progress))


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


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device):
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        if ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def maybe_compile_model(model, device):
    if not ENABLE_COMPILE or device.type != "cuda" or not hasattr(torch, "compile"):
        print("torch.compile: disabled")
        return model

    try:
        compiled_model = torch.compile(model, mode=COMPILE_MODE)
        print(f"torch.compile: enabled ({COMPILE_MODE})")
        return compiled_model
    except Exception as exc:
        print(f"torch.compile unavailable, using eager mode ({exc})")
        return model


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def build_adamw_optimizer(params, lr, betas, weight_decay, device):
    if device.type == "cuda" and BNB_AVAILABLE:
        print("Optimizer: bitsandbytes AdamW8bit")
        return bnb.optim.AdamW8bit(params, lr=lr, betas=betas, weight_decay=weight_decay)

    if BNB_IMPORT_ERROR is not None:
        print(f"bitsandbytes unavailable, using torch.optim.AdamW ({BNB_IMPORT_ERROR})")
    else:
        print("Using torch.optim.AdamW")
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def split_parameters(model):
    """
    Split model parameters into two groups:
      - Muon group: All 2D weight matrices inside decoder blocks
        (attention projections and MLP weights). Skip embeddings and LM head.
      - AdamW group: All 1D parameters (biases, RMSNorm gains),
        the Embedding table, and the LM Head.
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "decoder.layers" in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def get_curriculum_probs(tokens_seen):
    """Returns dataset probabilities based on 4-phase curriculum (3B limit)."""
    # Datasets: [TinyStories, Cosmo-Stories, Cosmo-v2, FineWeb-Edu, ProofWriter, TinyCodes]
    if tokens_seen < 50_000_000:
        # Phase 1 (0 to 50M tokens): Basic Language & Simple Logic
        return [0.45, 0.40, 0.00, 0.00, 0.15, 0.00], "Phase 1: Basic"
    elif tokens_seen < 1_500_000_000:
        # Phase 2 (50M to 1.5B tokens): Knowledge & Reasoning
        return [0.015, 0.015, 0.55, 0.25, 0.10, 0.07], "Phase 2: Knowledge"
    elif tokens_seen < 2_500_000_000:
        # Phase 3 (1.5B to 2.5B tokens): Applied Logic & Code
        return [0.00, 0.00, 0.25, 0.40, 0.15, 0.20], "Phase 3: Logic/Code"
    else:
        # Phase 4 (2.5B to 3.0B tokens): Cooldown & Consolidation
        return [0.00, 0.00, 0.10, 0.30, 0.30, 0.30], "Phase 4: Cooldown"


def safe_get_batch(iterator, dataloader, dataset_name):
    """Robust batch fetcher with shard exhaustion fallback."""
    try:
        return next(iterator), iterator
    except StopIteration:
        print(f"Dataset {dataset_name} exhausted. Restarting stream...")
        iterator = iter(dataloader)
        try:
            return next(iterator), iterator
        except StopIteration:
            raise RuntimeError(f"CRITICAL: Dataset {dataset_name} is empty or broken.")


def get_dataset_num_shards(dataset):
    candidates = [dataset, getattr(dataset, "_ex_iterable", None)]

    for candidate in candidates:
        if candidate is None:
            continue
        for attr in ("n_shards", "num_shards"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    return 1


def build_streaming_loader(dataset, tokenizer, device):
    num_shards = get_dataset_num_shards(dataset)
    num_workers = min(MAX_DATALOADER_WORKERS, num_shards)
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    dataloader = DataLoader(
        StreamingLanguageModelDataset(
            dataset,
            SEQ_LEN,
            tokenizer,
            text_batch_size=TOKENIZER_TEXT_BATCH_SIZE,
        ),
        **loader_kwargs,
    )
    return dataloader, num_workers, num_shards


def format_duration(seconds):
    if not math.isfinite(seconds):
        return "--:--:--"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_tokens(tokens):
    if tokens >= 1_000_000_000:
        return f"{tokens / 1_000_000_000:.2f}B"
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.2f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.2f}K"
    return str(tokens)


def train_mixed_strategy(model, optimizer_muon, optimizer_adamw, tokenizer, global_tracker=None):
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Total steps for AdamW LR schedule
    tokens_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM_STEPS
    total_steps = TOTAL_TRAINING_TOKENS // tokens_per_step

    # --- Dataset Loading and Standardization ---
    print("Loading curriculum datasets with streaming...")

    # 1. TinyStories
    ds_tiny = load_dataset("roneneldan/TinyStories", split="train", streaming=True).select_columns(["text"])
    
    # 2. Cosmopedia Stories (General Version)
    ds_stories = load_dataset("HuggingFaceTB/cosmopedia", name="stories", split="train", streaming=True).select_columns(["text"])
    
    # 3. Cosmopedia-v2
    ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", split="train", streaming=True).select_columns(["text"])
    
    # 4. FineWeb-Edu
    ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).select_columns(["text"])
    
    # 5. ProofWriter
    ds_proof = load_dataset("tasksource/proofwriter", split="train", streaming=True)
    ds_proof = ds_proof.map(lambda x: {"text": f"Context:\n{x['theory']}\n\nQuestion: {x['question']}\nAnswer: {str(x['answer'])}"}).select_columns(["text"])
    
    # 6. TinyCodes (nampdn-ai/tiny-codes)
    ds_codes = load_dataset("nampdn-ai/tiny-codes", split="train", streaming=True)
    ds_codes = ds_codes.filter(lambda x: x["programming_language"] == "Python")
    ds_codes = ds_codes.map(lambda x: {"text": f"Question:\n{x['prompt']}\n\nCode:\n{x['response']}"}).select_columns(["text"])

    datasets = [ds_tiny, ds_stories, ds_cosmo, ds_fineweb, ds_proof, ds_codes]
    dataset_names = ["TinyStories", "Cosmo-Stories", "Cosmo-v2", "FineWeb-Edu", "ProofWriter", "TinyCodes"]

    # --- DataLoaders and Iterators ---
    dataloaders = []
    dataloader_stats = []

    for dataset, dataset_name in zip(datasets, dataset_names):
        dataloader, num_workers, num_shards = build_streaming_loader(dataset, tokenizer, device)
        dataloaders.append(dataloader)
        dataloader_stats.append((dataset_name, num_workers, num_shards))

    iterators = [iter(dl) for dl in dataloaders]

    print("Streaming loaders:")
    for dataset_name, num_workers, num_shards in dataloader_stats:
        print(f"  - {dataset_name}: shards={num_shards}, workers={num_workers}")
    print("Runtime tuning: CUDA_LAUNCH_BLOCKING=off, batched tokenization=on, worker sharding=on")

    pbar = tqdm(
        total=TOTAL_TRAINING_TOKENS,
        dynamic_ncols=True,
        unit="tok",
        unit_scale=True,
        unit_divisor=1000,
    )
    loss_window = deque(maxlen=50)
    data_time_window = deque(maxlen=20)
    compute_time_window = deque(maxlen=20)

    lr_multiplier = 1.0

    optimizer_muon.zero_grad(set_to_none=True)
    optimizer_adamw.zero_grad(set_to_none=True)

    step = 0
    opt_step = 0
    current_adamw_lr = LR
    current_muon_lr = 0.02
    current_avg_loss = 0.0
    phase_name = "Phase 0"
    current_loss = float("nan")
    avg_data_ms = 0.0
    avg_compute_ms = 0.0
    bottleneck = "init"
    model.train()

    while global_tracker['tokens_seen'] < TOTAL_TRAINING_TOKENS:
        step += 1
        accum_step = ((step - 1) % GRAD_ACCUM_STEPS) + 1
        should_log = (
            step == 1
            or accum_step == GRAD_ACCUM_STEPS
            or step % LOG_EVERY_STEPS == 0
        )
        step_timer = time.perf_counter() if should_log else None

        # Curriculum Routing
        current_probs, phase_name = get_curriculum_probs(global_tracker['tokens_seen'])
        dataset_idx = random.choices(range(len(datasets)), weights=current_probs, k=1)[0]
        dataset_name = dataset_names[dataset_idx]
        
        # Safe Batch Fetching
        batch, iterators[dataset_idx] = safe_get_batch(iterators[dataset_idx], dataloaders[dataset_idx], dataset_names[dataset_idx])
        if should_log:
            data_time_window.append(time.perf_counter() - step_timer)

        input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
        targets = batch["targets"].to(device, non_blocking=(device.type == "cuda"))
        batch_tokens = input_ids.numel()
        remaining_before_step = max(0, TOTAL_TRAINING_TOKENS - global_tracker['tokens_seen'])

        # Mark step for CUDA graphs (if compile is re-enabled later)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()

        # Inlined forward + backward
        compute_timer = time.perf_counter() if should_log else None
        with get_autocast_context(device):
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        if accum_step == GRAD_ACCUM_STEPS:
            opt_step += 1
            
            
            # Update AdamW LR with multiplier
            current_adamw_lr = get_adamw_lr(opt_step, total_steps) * lr_multiplier
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = current_adamw_lr

            # Update Muon LR with multiplier
            current_muon_lr = 0.02 * lr_multiplier
            for param_group in optimizer_muon.param_groups:
                param_group['lr'] = current_muon_lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_muon.step()
            optimizer_adamw.step()

            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)

        if should_log:
            if device.type == "cuda":
                torch.cuda.synchronize()
            compute_time_window.append(time.perf_counter() - compute_timer)

        # Updates
        global_tracker['tokens_seen'] += batch_tokens
        pbar.update(min(batch_tokens, remaining_before_step))

        if should_log or global_tracker['tokens_seen'] >= TOTAL_TRAINING_TOKENS:
            current_loss = loss.detach().float().item()
            loss_window.append(current_loss)
            current_avg_loss = sum(loss_window) / len(loss_window)
            avg_data_ms = 1000 * (sum(data_time_window) / len(data_time_window)) if data_time_window else 0.0
            avg_compute_ms = 1000 * (sum(compute_time_window) / len(compute_time_window)) if compute_time_window else 0.0
            if avg_data_ms > avg_compute_ms * 1.1:
                bottleneck = "data"
            elif avg_compute_ms > avg_data_ms * 1.1:
                bottleneck = "gpu"
            else:
                bottleneck = "balanced"

            elapsed = max(1e-6, time.time() - global_tracker['start_time'])
            overall_tok_s = global_tracker['tokens_seen'] / elapsed
            eta_seconds = (TOTAL_TRAINING_TOKENS - global_tracker['tokens_seen']) / max(overall_tok_s, 1e-6)
            progress = " | ".join([
                phase_name,
                f"ds={dataset_name}",
                f"loss={current_loss:.4f}",
                f"avg={current_avg_loss:.4f}",
                f"ga={accum_step}/{GRAD_ACCUM_STEPS}",
                f"opt={opt_step:,}/{total_steps:,}",
                f"adam={current_adamw_lr:.2e}",
                f"muon={current_muon_lr:.2e}",
                f"tok={format_tokens(global_tracker['tokens_seen'])}/{format_tokens(TOTAL_TRAINING_TOKENS)}",
                f"tok/s={overall_tok_s:,.0f}",
                f"dl={avg_data_ms:.0f}ms",
                f"cmp={avg_compute_ms:.0f}ms",
                f"slow={bottleneck}",
                f"eta={format_duration(eta_seconds)}",
            ])
            pbar.set_postfix_str(progress, refresh=False)

    pbar.close()
    print(f"Training Complete. Total Tokens: {global_tracker['tokens_seen']:,}")


def train():
    device = resolve_device()
    configure_runtime(device)

    print(f"Using device: {device}")
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = Tokenizer.from_file("tokenizer.json")
    except Exception as exc:
        print(f"CRITICAL: Failed to load tokenizer.json ({exc})")
        return

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != VOCAB_SIZE:
        print(f"Tokenizer vocab size {vocab_size} overrides config VOCAB_SIZE={VOCAB_SIZE}.")

    model = get_model(vocab_size).to(device)

    # --- Parameter Grouping ---
    muon_params, adamw_params = split_parameters(model)
    
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"Model Parameters: {n_muon + n_adamw:,}")

    # --- Separate Dual Optimizers ---
    optimizer_muon = SingleDeviceMuon(muon_params, lr=0.02, momentum=0.95, nesterov=True)
    optimizer_adamw = build_adamw_optimizer(
        adamw_params,
        lr=LR,
        betas=(0.90, 0.95),
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    model = maybe_compile_model(model, device)

    # Global Progress Tracker
    global_tracker = {'start_time': time.time(), 'tokens_seen': 0}

    train_mixed_strategy(
        model=model,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        tokenizer=tokenizer,
        global_tracker=global_tracker
    )

    torch.save(unwrap_model(model).state_dict(), f"{MODEL_FOLDER}/model_final.pt")
    total_time = time.time() - global_tracker['start_time']
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE. Total Time: {total_time/3600:.2f} hours")
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
