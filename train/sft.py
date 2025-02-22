import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import json

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

@dataclass
class TrainingConfig:
    model_name: str = field(default=None)
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default=None)
    dagger: bool = field(default=False)

    def __post_init__(self):
        # Load config if not provided
        if self.model_name is None:
            config = load_config()
            model_key = config["model_choices"]["base"]
            self.model_name = config["models"][model_key]["hf_path"]
            self.block_size = config["models"][model_key]["max_length"]

        # Use provided train_file_path if available
        if self.train_file_path is None:
            raise ValueError("train_file_path must be provided")
        
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    import torch
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
        logging.info(f"Train split size: {len(dataset['train'])}")
        logging.info(f"Test split size: {len(dataset['test'])}")
        
        # Ensure dataset has the right structure
        if not isinstance(dataset, DatasetDict):
            raise ValueError(f"Dataset should be a DatasetDict but got {type(dataset)}")
        if not isinstance(dataset['train'], Dataset):
            raise ValueError(f"Train split should be a Dataset but got {type(dataset['train'])}")
        if 'text' not in dataset['train'].features:
            raise ValueError(f"Dataset missing 'text' field. Features: {dataset['train'].features}")

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
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

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    # Set distributed training configuration
    args.ddp_backend = 'nccl'  # Use NCCL for GPU communication
    args.distributed_state = None  # Let the Trainer handle distributed state
    
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    logging.critical(f'Outputting to: `{args.output_dir}`')
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
