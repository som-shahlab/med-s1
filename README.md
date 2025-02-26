# med-s1 Pipeline

The pipeline consists of three stages: curation, training, and evaluation. All experiments are defined in `results.json`, which tracks configuration and results for each stage.

## Setup

```bash
source config.sh  # Sets HF cache, output dirs, API keys
```

## Experiments

Experiments are defined in `results.json`. Each experiment has:
- Configuration for curation, training, and evaluation
- Results and metrics from each stage
- Consistent path handling using cleaned experiment names

Example experiments:
- med-s1-1k: Current configuration with 1k samples
- med-s1-5k: Larger dataset with 5k samples
- med-s1-25k: Full dataset without curation
- random-1k: Random sampling baseline
- base: Base LLaMA model without fine-tuning

## Pipeline Stages

### 1. Curation

The curation pipeline processes medical questions through:
1. Quality filtering: Remove examples with missing fields and exact 1024 token responses
2. Difficulty filtering: Remove examples that base model answers correctly
3. Diversity sampling: Select examples with long chain-of-thought reasoning (≥1000 tokens), balanced across specialties

```bash
# Run curation for an experiment
sbatch curate_med_s1k.sh <experiment_name>
```

Output files in `$MED_S1K_OUTPUT/<experiment_name>/`:
- `med_s1k_filtered.parquet`: Full dataset with filtering metadata
- `med_s1k_curated.parquet`: Selected examples only
- `med_s1k_formatted/`: HuggingFace dataset ready for training

### 2. Training

Train models using FSDP (Fully Sharded Data Parallel) and TRL (Transformer Reinforcement Learning):

```bash
# Train model for an experiment
sbatch train/sft_carina.sh <experiment_name>
```

Models are saved in `$CACHE_DIR/ckpts/<experiment_name>/`. The training uses:
- TRL's SFTTrainer for efficient fine-tuning
- FSDP for distributed training across GPUs
- Automatic checkpointing and state management
- Consistent experiment-based paths

### 3. Evaluation

Evaluate models using vllm for efficient inference:

```bash
# Run evaluation for an experiment
sbatch eval/eval.sh <experiment_name>
```

Outputs in `$CACHE_DIR/eval/<experiment_name>/`:
- `eval_results.json`: Overall metrics
- `eval_predictions.jsonl`: Raw model predictions

This takes ~5 minutes to run on all 8k eval examples in `eval/data/eval_data.json` for an 8B model on 1 H100 with batch_size=1024.
