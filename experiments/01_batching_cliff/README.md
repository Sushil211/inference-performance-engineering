# Experiment 01: The Batching Cliff

This experiment benchmarks the autoregressive generation throughput and VRAM consumption of Llama 3.1 8B across scaling batch sizes.

## Execution
To reproduce the experiment on a local GPU or cloud instance:

1. Ensure your environment is activated and dependencies are installed.
2. Adjust the batch sizes or output lengths in `config.yaml` if necessary.
3. Execute the benchmark suite:
    ```bash
    bash run.sh
    ```
4. Generate the visualization charts:
    ```bash
    python3 plot_vram_cliff.py
    ```

## Findings
For the full hardware analysis, VRAM fragmentation breakdown, and throughput graphs, please see the [**analysis.md**](./analysis.md) report.