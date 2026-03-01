import os
# Force Hugging Face to use the persistent vault BEFORE importing anything else
os.environ["HF_HOME"] = "/workspace/huggingface_cache"

import torch
import time
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
INPUT_LENGTH = 128  # Number of prompt tokens
OUTPUT_LENGTH = 50  # Number of tokens to generate
WARMUP_RUNS = 2
MEASUREMENT_RUNS = 3
RESULTS_FILE = "results/01_batching_cliff/batching_metrics_llama3_1_8b.csv"

def main():
    print(f"🚀 Loading {MODEL_ID} into VRAM...")
    
    # Load model in bfloat16 to maximize RTX 4090 Tensor Core efficiency
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    results = []

    print("\n📊 Starting Batching Benchmark...")
    print("-" * 60)
    print(f"{'Batch Size':<12} | {'Latency (ms)':<15} | {'Throughput (tok/s)':<20}")
    print("-" * 60)

    # Generate a dummy prompt of exact length
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=INPUT_LENGTH)
    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    with torch.no_grad():
        for batch_size in BATCH_SIZES:
            # Expand the single input into the desired batch size
            batch_input_ids = input_ids.expand(batch_size, -1)
            batch_attention_mask = attention_mask.expand(batch_size, -1)
            
            # --- WARMUP ---
            for _ in range(WARMUP_RUNS):
                _ = model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=OUTPUT_LENGTH,
                    min_new_tokens=OUTPUT_LENGTH,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # --- MEASUREMENT ---
            torch.cuda.synchronize() # Wait for all warmups to finish
            start_time = time.perf_counter()
            
            for _ in range(MEASUREMENT_RUNS):
                _ = model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=OUTPUT_LENGTH,
                    min_new_tokens=OUTPUT_LENGTH,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize() # Wait for all measurements to finish
            end_time = time.perf_counter()
            
            # --- METRICS CALCULATION ---
            total_time_sec = end_time - start_time
            avg_time_per_run_sec = total_time_sec / MEASUREMENT_RUNS
            avg_time_per_run_ms = avg_time_per_run_sec * 1000
            
            # Throughput = Total generated tokens / Total time
            total_generated_tokens = batch_size * OUTPUT_LENGTH
            throughput_tok_per_sec = total_generated_tokens / avg_time_per_run_sec
            
            print(f"{batch_size:<12} | {avg_time_per_run_ms:<15.2f} | {throughput_tok_per_sec:<20.2f}")
            
            results.append({
                "batch_size": batch_size,
                "latency_ms": round(avg_time_per_run_ms, 2),
                "throughput_tok_sec": round(throughput_tok_per_sec, 2)
            })

    # Save to CSV for the plotting script later
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["batch_size", "latency_ms", "throughput_tok_sec"])
        writer.writeheader()
        writer.writerows(results)
        
    print("-" * 60)
    print(f"✅ Benchmark complete. Data saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()