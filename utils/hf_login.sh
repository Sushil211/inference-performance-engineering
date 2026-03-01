#!/bin/bash

# Ensure Hugging Face routes large models to the persistent 200GB vault
export HF_HOME="/workspace/huggingface_cache"

# Trigger the Hugging Face authentication prompt
python -c "from huggingface_hub import login; login()"