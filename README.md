# med-s1 Curation Pipeline

The curation pipeline (`data/curate_med_s1k.py`) processes medical questions through:

1. Quality filtering: Remove examples with missing fields
2. Difficulty filtering: Remove examples that base model answers correctly
3. Diversity sampling: Select examples with long chain-of-thought reasoning (≥1000 tokens), balanced across specialties

Output files in `$MED_S1K_OUTPUT/[version]_[timestamp]/`:
- `med_s1k_filtered.parquet`: Full dataset with filtering metadata
- `med_s1k_curated.parquet`: Selected examples only
- `med_s1k_formatted/`: HuggingFace dataset ready for training

Setup requires:
```bash
source config.sh  # Sets HF cache, output dirs, API keys
```

See below for training details.


## Guide

### 1. Setup Configs

First, set up your environment vars.
```bash
source config.sh
```

Second, make sure the `config.json` is set up correctly for your experiments.

```bash
# Edit config.json
vim config.json
```

Contents of `config.json`:
```json
{
    "model_choices": {
        "base": "llama3.1:8b",
        "specialty_labeler": "gemini-2.0-flash",
        "base_judge": "gemini-2.0-flash"
    },
    "models": {
        "llama3.1:8b": {
            "hf_path": "meta-llama/Llama-3.1-8B-Instruct",
            "max_length": 128000
        },
        "gpt4o-mini": {
            "max_length": 128000,
            "pricing": {
                "input": 0.15,
                "output": 0.60
            }
        },
        "gemini-2.0-flash": {
            "max_length": 128000,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "pricing": {
                "input": 0.10,
                "output": 0.40,
                "rpm": 2000
            }
        }
    },
    "curation": {
        "llama_batch_size": 1024, // How many examples to process at once for the base model
        "gemini_batch_size": 512, // How many examples to process at once for Gemini
        "initial_sample": 0, // Randomly samples `x` examples from the HuaotoGPT training dataset before running the curation pipeline. If set to 0, will run on the entire dataset. Useful for debugging.
        "version": "plumbing_test_001" // Unique string for output files
    }
}
```

### 2. Generate med-s1k Dataset

Next, we can generate the med-s1k dataset. This reads in the original HuaotoGPT training dataset with 25k examples (`data/huatuogpt_o1_train_questions/`) and filters it down to 1k samples using Caleb's s1-based filtering pipeline with these three steps:

1. **Quality filtering:** Remove examples with missing fields
2. **Difficulty filtering:** Remove examples that base model answers correctly
3. **Diversity sampling:** Select examples with long chain-of-thought reasoning (≥1000 tokens), balanced across specialties

The output is a directory (`$MED_S1K_OUTPUT/[version]_[timestamp]/`) with the following files:
- `med_s1k_filtered.parquet`: Full 25k dataset with filtering annotations for each example (e.g. whether it was filtered out, and why)
- `med_s1k_curated.parquet`: Selected 1k examples only
- `med_s1k_formatted/`: The 1k examples, stored as a HuggingFace dataset ready for training. This formats the examples into chat conversations with think/answer markers.

```bash
# Run script directly
python3 data/curate_med_s1k.py

# Or... run script with sbatch
sbatch curate_med_s1k.sh
```

### 3. Train Models

Train your model on the med-s1k dataset.

```bash
sbatch train/sft_carina.sh
```

## Eval

Note: SGLang only work works with CUDA 12.1.

Run eval on **HuatuoGPT-o1-8B** model:

```bash
# Start SGLang server
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28035
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1

# Run evals
python eval/eval.py --model_name $model_name  --path_to_eval_json eval/data/eval_data.json --port $port --path_to_output_dir eval/results

# Free up GPU memory
bash eval/kill_sglang_server.sh
```

Run eval on **local model**:

```bash
# Start SGLang server
model_name="/share/pi/nigam/users/calebwin/hf_cache/ckpts/med_s1_/share/pi/nigam/data/med_s1k/s1_replication/med_s1k_formatted_bs8_lr1e-5_epoch5_wd1e-4_20250220_011621/checkpoint-450"
port=28035
python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1

# Run evals
python eval/eval.py --model_name $model_name  --path_to_eval_json eval/data/eval_data.json --port $port --path_to_output_dir eval/results

# Free up GPU memory
bash eval/kill_sglang_server.sh
```

These scripts will output two `.json` files in the `eval/results` directory, one containing the raw predictions and the other containing the overall scores. 

This should take ~5 minutes to run on all 8k eval examples in `eval/data/eval_data.json` for an 8B model on 1 H100 with batch_size=1024.