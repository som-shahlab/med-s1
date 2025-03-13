"""Main training script for medical dialogue model fine-tuning.

This script handles:
1. Argument parsing for training configuration
2. Setting up the training environment
3. Running the training process

The training uses DeepSpeed ZeRO-2 optimization and is configurable to work with
different numbers of H100 GPUs (typically 2 or 4). The gradient accumulation steps
are automatically scaled based on the number of GPUs to maintain the same total batch size.
Checkpoints are saved after each epoch, with the final model saved directly in
the output directory (not in a 'final' subdirectory).

Example:
    $ python -m train.sft \
        --experiment_name="experiment-name" \
        --results_json="/path/to/results.json" \
        --model_name="model-name" \
        --train_file_path="/path/to/data" \
        --output_dir="/path/to/output"
"""

import os
import argparse
import logging
from transformers import set_seed

from train.config import TrainingConfig
from train.trainer import SFTTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Training arguments')
    
    # Required arguments
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment from results.json')
    parser.add_argument('--results_json', type=str, required=True,
                      help='Path to results.json containing experiment configuration')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the model to fine-tune')
    parser.add_argument('--train_file_path', type=str, required=True,
                      help='Path to the training data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save model checkpoints')
    
    # Optional arguments with defaults
    parser.add_argument('--block_size', type=int, default=8192,
                      help='Maximum sequence length')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                      help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                      help='Number of steps to accumulate gradients')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                      help='Weight decay coefficient')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                      help='Ratio of warmup steps to total steps')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                      help='Adam beta1 parameter')
    parser.add_argument('--adam_beta2', type=float, default=0.95,
                      help='Adam beta2 parameter')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                      help='Adam epsilon parameter')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with reduced dataset')
    parser.add_argument('--max_ckpts', type=int, default=2,
                      help='Maximum number of checkpoints to keep')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                      help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.01,
                      help='Minimum change in the monitored quantity to qualify as an improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='loss', choices=['loss', 'accuracy'],
                      help='Metric to monitor for early stopping')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Log configuration
    logger.info("Starting training with configuration:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Create config
    config = TrainingConfig(**vars(args))
    
    # Initialize and run trainer
    trainer = SFTTrainer(config)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()
