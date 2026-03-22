import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
import os
import sys

from config import *
from model import build_llama

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the distribution
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep tokens
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def generate(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=50, 
    temperature=1.0, 
    top_k=0, 
    top_p=0.9, 
    device="cuda",
    stream=True
):
    model.eval()
    
    # Tokenize prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], device=device)
    
    # Initial forward pass to populate KV cache
    logits, past_kv = model(input_ids, use_cache=True)
    
    # Process only the last logit
    next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
    
    if temperature == 0:
        # Greedy search
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    else:
        # Top-k / Top-p filtering
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    
    generated_tokens = [next_token.item()]
    
    if stream:
        print(tokenizer.decode([next_token.item()]), end="", flush=True)

    # Autoregressive generation loop using KV cache
    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.token_to_id("<EOS>"): # Assuming <EOS> is used
             break
             
        logits, past_kv = model(next_token, past_kv_list=past_kv, use_cache=True)
        next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        
        if temperature == 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        generated_tokens.append(next_token.item())
        if stream:
            print(tokenizer.decode([next_token.item()]), end="", flush=True)
            
    if stream:
        print()
        
    return tokenizer.decode(generated_tokens)

def main():
    parser = argparse.ArgumentParser(description="Generate text using FlashLLaMA")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for generation (if None, enters interactive mode)")
    parser.add_argument("--weights", type=str, default="checkpoints/model_final.pt", help="Path to model weights")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, or cpu)")
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    print(f"Using device: {device}")
    
    # Load tokenizer
    if not os.path.exists("tokenizer.json"):
        print("Error: tokenizer.json not found.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Build model
    vocab_size = tokenizer.get_vocab_size()
    model = build_llama(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=N_LAYERS,
        num_q_heads=N_Q_HEADS,
        num_kv_heads=N_KV_HEADS,
        d_ff=D_FF,
        dropout=0.0 # No dropout for inference
    ).to(device)
    
    # Load weights
    if not os.path.exists(args.weights):
        # Check if the weight is in the local dir instead of checkpoints/
        if os.path.exists("model_final.pt"):
            args.weights = "model_final.pt"
        else:
            print(f"Error: Weights file not found at {args.weights}")
            sys.exit(1)
            
    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device)
    # Handle possible compiled model prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    if args.prompt:
        print("\nPrompt:", args.prompt)
        print("Response: ", end="")
        generate(
            model, 
            tokenizer, 
            args.prompt, 
            max_new_tokens=args.max_tokens, 
            temperature=args.temp, 
            top_k=args.top_k, 
            top_p=args.top_p, 
            device=device,
            stream=True
        )
    else:
        print("\n--- FlashLLaMA Interactive Mode ---")
        print("Type 'exit' or 'quit' to stop, or press Ctrl+C.\n")
        while True:
            try:
                user_prompt = input("Prompt >>> ")
                if user_prompt.lower() in ["exit", "quit"]:
                    break
                if not user_prompt.strip():
                    continue
                
                print("Response: ", end="")
                generate(
                    model, 
                    tokenizer, 
                    user_prompt, 
                    max_new_tokens=args.max_tokens, 
                    temperature=args.temp, 
                    top_k=args.top_k, 
                    top_p=args.top_p, 
                    device=device,
                    stream=True
                )
                print("-" * 40)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
        print("\nExiting interactive mode.")

if __name__ == "__main__":
    main()
