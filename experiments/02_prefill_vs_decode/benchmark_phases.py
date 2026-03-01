import os
# Force Hugging Face to use the persistent vault BEFORE importing anything else
os.environ["HF_HOME"] = "/workspace/huggingface_cache"

import torch
import time
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Create an artificially massive prompt to stress the Prefill phase
PROMPT_CHUNK = "Explain the complete architecture of a modern GPU, including Tensor Cores, VRAM, and memory bandwidth. "
PROMPT = PROMPT_CHUNK * 30  

def main():
    print(f"🚀 Loading {MODEL_ID} into VRAM...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    print("\n📊 Tokenizing prompt...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[1]
    print(f"Prompt Length: {prompt_length} tokens")
    
    # Use a Streamer to catch the exact millisecond the first token pops out
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=50, # Generate 50 words
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    print("-" * 60)
    print("⚡ Starting Generation (Watch the phases)...")
    
    # Run generation in a background thread so our main thread can time the tokens
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    
    start_time = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    
    thread.start()

    # The loop blocks until the next token is ready
    for _ in streamer:
        current_time = time.perf_counter()
        
        # If this is the very first token, capture TTFT!
        if first_token_time is None:
            first_token_time = current_time
            ttft_ms = (first_token_time - start_time) * 1000
            print(f"⏱️  Prefill Phase (TTFT): {ttft_ms:.2f} ms to process {prompt_length} tokens")
            decode_start_time = current_time
        else:
            tokens_received += 1

    thread.join()
    end_time = time.perf_counter()

    # --- Metrics Calculation ---
    total_decode_time = end_time - decode_start_time
    avg_itl_ms = (total_decode_time / tokens_received) * 1000
    decode_throughput = tokens_received / total_decode_time

    print(f"⏱️  Decode Phase (ITL):  {avg_itl_ms:.2f} ms per token")
    print(f"🏎️  Decode Throughput:   {decode_throughput:.2f} tokens/sec")
    print("-" * 60)
    print("✅ Experiment Complete.")

if __name__ == "__main__":
    main()