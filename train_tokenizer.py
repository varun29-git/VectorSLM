
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.trainers import BpeTrainer

def train_tokenizer():
    print("Loading dataset samples for tokenizer training...")
    # 1. Cosmopedia
    ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", "web_samples_v2", split="train", streaming=True)
    # 2. FineWeb-Edu
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    # 3. Evol-Instruct-Code (20%)
    try:
        ds_code = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
    except Exception:
        # Fallback if streaming fails or dataset issues (mostly for robustness)
        ds_code = [] 

    # Create an iterator that yields text from all three (Mix: 50/30/20)
    def batch_iterator(batch_size=1000):
        buffer = []
        
        # Take 10000 examples from Cosmo (50%)
        print("Sampling Cosmopedia (10k)...")
        for i, item in enumerate(ds_cosmo):
            if i >= 10000: break
            buffer.append(item['text'])

        # Take 6000 examples from FW-Edu (30%)
        print("Sampling FineWeb-Edu (6k)...")
        for i, item in enumerate(ds_fw):
            if i >= 6000: break
            buffer.append(item['text'])
            
        # Take 4000 examples from Code (20%)
        print("Sampling Evol-Instruct (4k)...")
        for i, item in enumerate(ds_code):
            if i >= 4000: break
            # Combine instruction and output
            text = f"{item.get('instruction', '')}\n{item.get('output', '')}"
            buffer.append(text)
            
        print(f"Collected {len(buffer)} samples. Training tokenizer...")
        
        for i in range(0, len(buffer), batch_size):
            yield buffer[i : i + batch_size]

    # Initialize Tokenizer (BPE)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=16384,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Save
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")
    
    # Verification
    print("\nVerification:")
    output = tokenizer.encode("Hello, this is a test.")
    print(f"Encoded 'Hello, this is a test.': {output.ids}")
    print(f"Decoded: {tokenizer.decode(output.ids)}")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_tokenizer()
