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
    block_size: int = field(default=4096)
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
            self.block_size = 4096

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
    
    # Debug collator to monitor token IDs and text
    def debug_collator(features):
        # # Debug what's in features
        # logging.warning(f"\nFeatures type: {type(features)}")
        # if features:
        #     logging.warning(f"First feature type: {type(features[0])}")
        #     if isinstance(features[0], dict):
        #         logging.warning(f"Feature keys: {features[0].keys()}")
        #         if 'text' in features[0]:
        #             logging.warning(f"\nRaw text sample: {features[0]['text'][:200]}")
        #         else:
        #             logging.warning("No 'text' key found in features")
        #     else:
        #         logging.warning(f"Feature content: {features[0]}")

        # Apply collation
        batch = collator(features)

        # Monitor token IDs and text
        if "input_ids" in batch:
            all_ids = batch["input_ids"].flatten().tolist()
            min_id, max_id = min(all_ids), max(all_ids)
            logging.warning(f"Token ID range: min={min_id}, max={max_id} out of {len(tokenizer)}")

            # Show first example's text
            logging.warning("\nFirst example decoded:")
            logging.warning(tokenizer.decode(batch["input_ids"][0][:150]))

            # Log label stats and text
            if "labels" in batch:
                labels = batch["labels"].flatten().tolist()
                mask_ratio = labels.count(-100) / len(labels)
                logging.warning(f"Label masking ratio: {mask_ratio:.2f}")

                # Show what we're actually training on (non -100 labels)
                first_labels = batch["labels"][0].tolist()
                training_tokens = [
                    batch["input_ids"][0][i].item()
                    for i, label in enumerate(first_labels)
                    if label != -100
                ]
                # if training_tokens:
                #     logging.warning("\nTraining on text:")
                #     logging.warning(tokenizer.decode(training_tokens))

        return batch

    args.dataset_text_field = 'text'
    # args.max_grad_norm = 1.0  # Add gradient clipping
    
    # Set distributed training configuration
    args.ddp_backend = 'nccl'
    args.distributed_state = None
    
    # Create trainer with debug collator
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=debug_collator,
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
    
    # Save final model
    logging.info("Saving model...")
    trainer.save_model()


if __name__ == "__main__":
    train()
