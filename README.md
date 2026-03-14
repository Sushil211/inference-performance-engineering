# Inference Performance Engineering

A hands-on exploration of LLM inference bottlenecks, GPU memory behavior, and scheduling strategies in modern AI serving systems.

This repository contains a series of controlled experiments designed to understand the physical limits of Large Language Model inference and develop strategies to improve throughput, latency stability, and hardware utilization. 

The project focuses on the core engineering question:
**Why do LLM inference systems slow down and how can we design schedulers and memory managers that keep them stable under load?**

## Experiment Roadmap

### Phase 1 — The Physics of Serving
Understanding the hardware bottlenecks of LLM inference and standard PyTorch limitations.

* **[Experiment 01: The Batching Cliff & Memory Wall](./experiments/01_batching_cliff/)**
  * *Summary:* Profiled Llama 3.1 8B on an RTX 4090 to map the exact transition from compute-bound to memory-bound inference. Proved that contiguous KV Cache allocation wastes ~5GB of VRAM at Batch 128, hitting the High Bandwidth Memory (HBM) wall and causing latency to explode.

## Engineering Goal
The objective of this repository is to build a portfolio of systems experiments demonstrating the ability to identify hardware bottlenecks, measure VRAM utilization at the C++/CUDA level, and design serving policies that stabilize latency under heavy concurrent load.