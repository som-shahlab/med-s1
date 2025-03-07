import os
import json
import torch
import logging
import argparse
import datetime
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from tqdm import tqdm
import warnings
import wandb
from accelerate import Accelerator
import torch.distributed as dist

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

class PreformattedDataset(TorchDataset):
    def __init__(self, data_path, tokenizer, block_size, debug=False):
        # Load dataset
        logging.info(f"Loading dataset from: {data_path}")
        dataset = load_from_disk(data_path)
        
        # Handle debug mode with reduced samples
        if debug:
            min_samples = 32  # At least 8 samples per GPU
            dataset = DatasetDict({
                'train': Dataset.from_dict(dataset['train'][:min_samples]),
                'test': Dataset.from_dict(dataset['test'][:min_samples])
            })
            logging.warning(f"Debug mode: Using {min_samples} samples")
            logging.warning(f"Train split size: {len(dataset['train'])}")
            logging.warning(f"Test split size: {len(dataset['test'])}")
            logging.warning(f"First example: {dataset['train'][0]}")
        
        self.dataset = dataset['train']
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.debug = 0
        
    def __getitem__(self, idx):
        # Get pre-formatted text
        text = self.dataset[idx]['text']
        
        # Verify format
        assert text.startswith('<|begin_of_text|>'), f"Text doesn't start with begin_of_text: {text[:50]}"
        assert '<|eot_id|>' in text, f"Text doesn't contain eot_id: {text[-50:]}"
        
        return text
        
    def __len__(self):
        return len(self.dataset)
    
    def get_query_and_response(self, text):
        # Find the assistant section which contains the response
        assistant_start = text.find('<|start_header_id|>assistant<|end_header_id|>')
        assert assistant_start != -1, "No assistant section found"
        
        # Query is everything before assistant's response
        query = text[:assistant_start]
        # Response is everything after, including the assistant header
        response = text[assistant_start:]
        
        # Verify format
        assert query.startswith('<|begin_of_text|>'), f"Query doesn't start with begin_of_text: {query[:50]}"
        assert '<|eot_id|>' in query, f"Query doesn't contain eot_id: {query[-50:]}"
        assert '<|eot_id|>' in response, f"Response doesn't contain eot_id: {response[-50:]}"
        
        return query, response
    
    def collate_fn(self, batch):
        input_ids_list = []
        labels_list = []
        
        for text in batch:
            # Split into query and response
            query, response = self.get_query_and_response(text)
            
            # Encode full text
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Encode query to get its length
            query_ids = self.tokenizer.encode(query, add_special_tokens=False)
            
            # Create labels: -100 for query tokens, actual ids for response
            labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
            
            # Verify lengths match
            assert len(labels) == len(input_ids), f"Length mismatch: labels={len(labels)}, input_ids={len(input_ids)}"
            
            # Truncate to block size
            input_ids = input_ids[-self.block_size:]
            labels = labels[-self.block_size:]
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Get max length for padding
        max_length = max(len(ids) for ids in input_ids_list)
        max_length = min(max_length, self.block_size)
        
        # Pad sequences
        input_ids_list = [ids + [self.tokenizer.eos_token_id] * (max_length - len(ids)) for ids in input_ids_list]
        labels_list = [labs + [-100] * (max_length - len(labs)) for labs in labels_list]
        
        return {
            "input_ids": torch.LongTensor(input_ids_list),
            "labels": torch.LongTensor(labels_list)
        }

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        right = self.right.clone()
        total = self.total.clone()
        total_loss = self.total_loss.clone()
        
        if dist.is_initialized():
            dist.all_reduce(right, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
            dist.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (right / total).item() if total.item() > 0 else 0
        loss = total_loss.item() / (self.world_size * self.n_step) if self.n_step > 0 else 0

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
            
        return acc, loss

@dataclass
class TrainingConfig:
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
    debug: bool = field(default=False)
    max_ckpts: int = field(default=2)

def train():
    parser = argparse.ArgumentParser(description='Training arguments')
    # Required arguments
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_file_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Optional arguments with defaults
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_ckpts', type=int, default=2)
    
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))
    
    # Set seed
    set_seed(config.seed)
    
    # Initialize accelerator with DeepSpeed
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    # Initialize wandb on main process
    if accelerator.is_main_process:
        wandb.init(
            project="med-s1",
            entity="ehr-fm",
            config=asdict(config),
            group=config.experiment_name,
            name=f"{config.experiment_name}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    # Setup optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon
    )

    # Create dataset and dataloader
    train_dataset = PreformattedDataset(
        config.train_file_path,
        tokenizer,
        config.block_size,
        config.debug
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * config.warmup_ratio)

    # Create scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize metric tracker
    metric = SFTMetric(device=accelerator.device)

    # Print training info
    if accelerator.is_main_process:
        print("\nTraining Configuration:")
        print(f"  Number of GPUs: {accelerator.num_processes}")
        print(f"  Batch size per device: {config.per_device_train_batch_size}")
        print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
        print(f"  Total batch size: {config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}")
        print(f"  Total optimization steps: {max_train_steps}")
        print(f"  Number of warmup steps: {num_warmup_steps}")

    # Training loop
    total_batch_size = config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, config.num_train_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                # Update metrics
                metric(outputs.logits, batch["labels"], loss)
                acc, train_loss = metric.get_metric()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                if accelerator.is_main_process:
                    progress_bar.set_postfix(
                        loss=f"{train_loss:.3f}",
                        acc=f"{acc:.3f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    
                    wandb.log({
                        'loss': train_loss,
                        'accuracy': acc,
                        'learning_rate': lr_scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'step': completed_steps,
                    })

            if completed_steps >= max_train_steps:
                break

        # Save checkpoint at end of each epoch
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Get model state dict
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                checkpoint_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Rotate old checkpoints
            checkpoint_dirs = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
            if len(checkpoint_dirs) > config.max_ckpts:
                oldest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[0]
                oldest_checkpoint_path = os.path.join(config.output_dir, oldest_checkpoint)
                if os.path.exists(oldest_checkpoint_path):
                    shutil.rmtree(oldest_checkpoint_path)

    # Save final model
    if accelerator.is_main_process:
        print("\nSaving final model...")
        final_dir = os.path.join(config.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            final_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        tokenizer.save_pretrained(final_dir)
        
        wandb.finish()

if __name__ == "__main__":
    train()
