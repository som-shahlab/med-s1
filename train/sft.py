import wandb
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import warnings
import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import json
import torch
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

class CustomTrainer(trl.SFTTrainer):
    def log(self, logs):
        """
        Intercept SFTTrainer log method and log to wandb
        
            ```
            logs = {'loss': 2.1122, 'grad_norm': 11.75, 'learning_rate': 6.993006993006993e-08, 'epoch': 0.00070074015679061}
            ```
        """
        super().log(logs)  # Call the original log method

@dataclass
class TrainingConfig:
    model_name: str = field(default=None)
    block_size: int = field(default=4096)
    wandb_project: Optional[str] = "med-s1"
    wandb_entity: Optional[str] = "ehr-fm"
    train_file_path: Optional[str] = field(default=None)
    dagger: bool = field(default=False)

    def __post_init__(self):
        # Load config if not provided
        if self.model_name is None:
            config = load_config()
            model_key = config["model_choices"]["base"]
            self.model_name = config["models"][model_key]["hf_path"]
            self.block_size = 4096

        # Use provided train_file_path if available
        if self.train_file_path is None:
            raise ValueError("train_file_path must be provided")
        
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    
    # wandb
    wandb.init(entity="ehr-fm", project="med-s1")

    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Initialize CUDA and distributed setup
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize CUDA device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Set device and ensure it's initialized
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.init()
    
    # Wait for all devices to be ready
    torch.cuda.synchronize()
    
    # loading model
    kwargs = {
        "device_map": {"": local_rank},
        "torch_dtype": torch.bfloat16,
        "use_cache": False,
        "low_cpu_mem_usage": True
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **kwargs
    )

    # Load dataset from either HuggingFace hub or local disk
    if ':' in config.train_file_path:  # HuggingFace hub dataset (e.g. 'dataset:split')
        dataset = load_dataset(config.train_file_path)
    else:  # Local disk
        from datasets import load_from_disk, Dataset, DatasetDict
        logging.info(f"Loading dataset from: {config.train_file_path}")
        dataset = load_from_disk(config.train_file_path)
        
        # Log basic dataset info
        logging.warning(f"Train split size: {len(dataset['train'])}")
        logging.warning(f"Test split size: {len(dataset['test'])}")
        
        # Check dataset format
        logging.warning(f"Train features: {dataset['train'].features}")
        logging.warning(f"First example: {dataset['train'][0]}")
        
        # Ensure dataset has the right structure
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"Dataset should be a DatasetDict but got {type(dataset)}")
        if not isinstance(dataset['train'], Dataset):
            raise ValueError(f"Train split should be a Dataset but got {type(dataset['train'])}")
        if 'text' not in dataset['train'].features:
            raise ValueError(f"Dataset missing 'text' field. Features: {dataset['train'].features}")

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    print("Tokenizer length:", len(tokenizer))
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        raise ValueError(f"Model name {config.model_name} not supported")

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    args.dataset_text_field = 'text'
    args.max_grad_norm = 1.0  # Add gradient clipping
    
    # Set distributed training configuration
    args.ddp_backend = 'nccl'
    args.distributed_state = None
    
    # Logging
    args.report_to = [ "wandb" ]
    args.logging_dir = "./logs"
    args.logging_strategy = "steps"
    args.logging_steps = 1  # Log every 1 steps
    
    # Log config
    wandb.config.update(log_config)

    # Create trainer with debug collator
    trainer = CustomTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
        max_seq_length=config.block_size,
    )

    logging.info(f"[Rank {local_rank}] After loading model: "
             f"Allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, "
             f"Reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
    if local_rank == 0:
        for name, param in model.named_parameters():
            logging.debug(f"{name} dtype = {param.dtype}")
            break

    logging.critical(f'Outputting to: `{args.output_dir}`')
    
    # Synchronize all processes before training
    trainer.accelerator.wait_for_everyone()
    
    trainer.train()
    
    # # Ensure all processes are ready to save
    trainer.accelerator.wait_for_everyone()
    
    # Save final model (all ranks need to save their shards)
    local_rank = trainer.args.local_rank
    logging.info(f"[Rank {local_rank}] Saving model...")
    
    # Log existing files
    if os.path.exists(args.output_dir):
        logging.info(f"[Rank {local_rank}] Existing files: {os.listdir(args.output_dir)}")
        
        # Clean up any existing shard files to avoid FSDP save conflicts
        if trainer.args.save_strategy == "no":
            for f in os.listdir(args.output_dir):
                if f.startswith("model-") and f.endswith(".safetensors"):
                    try:
                        os.remove(os.path.join(args.output_dir, f))
                        logging.info(f"[Rank {local_rank}] Removed existing shard file: {f}")
                    except OSError as e:
                        logging.warning(f"[Rank {local_rank}] Failed to remove {f}: {e}")
    
    # Ensure all ranks are synced before saving
    trainer.accelerator.wait_for_everyone()
    
    trainer.save_model()
    
    # Log files after save
    if os.path.exists(args.output_dir):
        logging.info(f"[Rank {local_rank}] Files after save: {os.listdir(args.output_dir)}")

    # Final sync point
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
