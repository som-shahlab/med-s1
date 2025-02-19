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

        # Get latest versioned dataset from environment variable
        med_s1k_dir = os.environ.get('MED_S1K_OUTPUT')
        if not med_s1k_dir:
            raise ValueError("MED_S1K_OUTPUT environment variable not set")
        
        # List all version directories and sort by timestamp
        version_dirs = [d for d in os.listdir(med_s1k_dir) if d.startswith('plumbing_test_')]
        if not version_dirs:
            raise ValueError(f"No version directories found in {med_s1k_dir}")
            
        # Sort by timestamp (part after plumbing_test_)
        latest_dir = sorted(version_dirs, key=lambda x: x.split('_', 3)[3])[-1]
        formatted_path = os.path.join(med_s1k_dir, latest_dir, 'med_s1k_formatted')
        if not os.path.exists(formatted_path):
            raise ValueError(f"No formatted dataset found at {formatted_path}")
        self.train_file_path = formatted_path
        
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
    
    # loading model
    kwargs = {"device_map": "auto", "torch_dtype": "auto",
              "attn_implementation": "flash_attention_2", "use_cache": False}
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
    
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
