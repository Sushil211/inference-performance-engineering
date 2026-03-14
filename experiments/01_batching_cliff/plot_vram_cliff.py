import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(csv_path, output_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Set up the figure with 2 subplots (stacked vertically)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("The Batching Cliff & Memory Wall: Llama 3.1 8B on RTX 4090", fontsize=16, fontweight='bold')

    # --- TOP PLOT: Latency vs Throughput ---
    color1 = 'tab:red'
    ax1.set_ylabel('Latency (ms)', color=color1, fontweight='bold')
    ax1.plot(df['batch_size'].astype(str), df['latency_ms'], marker='o', color=color1, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Throughput (tok/s)', color=color2, fontweight='bold')
    ax2.plot(df['batch_size'].astype(str), df['throughput_tok_sec'], marker='s', color=color2, linewidth=2, label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.set_title("Compute vs Memory Bound Transition")

    # --- BOTTOM PLOT: VRAM Consumption ---
    color3 = 'tab:purple'
    ax3.set_xlabel('Batch Size', fontweight='bold')
    ax3.set_ylabel('Peak VRAM (GB)', color=color3, fontweight='bold')
    
    # Bar chart for VRAM to clearly show the capacity fill
    ax3.bar(df['batch_size'].astype(str), df['peak_vram_gb'], color=color3, alpha=0.7, width=0.5)
    
    # Add a horizontal line for the RTX 4090 physical limit
    ax3.axhline(y=24.0, color='black', linestyle='-.', linewidth=2, label='RTX 4090 VRAM Limit (24GB)')
    ax3.set_ylim(0, 26)
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.legend(loc='upper left')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    ax3.set_title("KV Cache Memory Fragmentation (Standard PyTorch)")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Graph saved to {output_path}")

if __name__ == "__main__":
    plot_metrics(
        "results/01_batching_cliff/batching_metrics_llama3_1_8b_vram.csv", 
        "results/01_batching_cliff/batching_cliff_vram_llama3.png"
    )