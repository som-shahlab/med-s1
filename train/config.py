"""Configuration for training run."""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for training run.
    
    Handles all hyperparameters and settings for a training run.
    """
    
    # Required arguments
    experiment_name: str = field(default=None)
    results_json: str = field(default=None)
    model_name: str = field(default=None)
    train_file_path: str = field(default=None)
    output_dir: str = field(default=None)
    
    # Training hyperparameters
    block_size: int = field(default=4096)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.05)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    num_train_epochs: int = field(default=3)
    
    # Other settings
    seed: int = field(default=42)
    debug: bool = field(default=False)  # Explicitly set default to False
    max_ckpts: int = field(default=2)
    validation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps between validation runs. If None, validates every 10% of an epoch"}
    )
    
    # Early stopping parameters
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=3)
    early_stopping_threshold: float = field(default=0.01)
    early_stopping_metric: str = field(default="loss")  # Options: "loss", "accuracy"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.experiment_name is None:
            raise ValueError("experiment_name must be provided")
        if self.results_json is None:
            raise ValueError("results_json must be provided")
        if self.model_name is None:
            raise ValueError("model_name must be provided")
        if self.train_file_path is None:
            raise ValueError("train_file_path must be provided")
        if self.output_dir is None:
            raise ValueError("output_dir must be provided")
        
        # Log debug mode
        if self.debug:
            import logging
            logging.warning("Debug mode is enabled in configuration")