"""Trainer class for supervised fine-tuning."""

import os
import logging
import datetime
import shutil
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from accelerate import Accelerator
from tqdm import tqdm
import wandb

from .data_utils import PreformattedDataset
from .metrics import SFTMetric
from .config import TrainingConfig

logger = logging.getLogger(__name__)

class SFTTrainer:
    """Trainer class for Supervised Fine-Tuning."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        if self.accelerator.is_main_process:
            wandb.init(
                project="med-s1",
                entity="ehr-fm",
                config=asdict(config),
                group=config.experiment_name,
                name=f"{config.experiment_name}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup(self):
        """Set up training components."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            trust_remote_code=True,
            padding_side="right"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True
        )
        self.model.gradient_checkpointing_enable()

        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )

        # Create dataset and dataloader
        train_dataset = PreformattedDataset(
            self.config.train_file_path,
            self.tokenizer,
            self.config.block_size,
            self.config.debug
        )
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn
        )

        # Calculate training steps accounting for distributed training
        world_size = self.accelerator.num_processes
        dataloader_length = len(self.train_dataloader)
        grad_acc_steps = self.config.gradient_accumulation_steps
        num_epochs = self.config.num_train_epochs
        
        # Calculate steps consistently - multiply first, then divide
        self.num_training_steps = int(dataloader_length * num_epochs) // grad_acc_steps // world_size
        
        # Log raw values before calculation
        if self.accelerator.is_main_process:
            logger.info("\nTraining Steps Calculation:")
            logger.info(f"  Raw dataloader length: {dataloader_length}")
            logger.info(f"  Number of epochs: {num_epochs}")
            logger.info(f"  Gradient accumulation steps: {grad_acc_steps}")
            logger.info(f"  World size (num GPUs): {world_size}")
            
            # Show calculation using same order as num_training_steps
            total_samples = dataloader_length * num_epochs
            steps_after_grad_acc = total_samples // grad_acc_steps
            steps_per_gpu = steps_after_grad_acc // world_size
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Steps after grad acc: {steps_after_grad_acc}")
            logger.info(f"  Steps per GPU: {steps_per_gpu}")
            logger.info(f"  Total steps: {self.num_training_steps}")
            
            # Show effective batch sizes
            per_gpu_batch = self.config.per_device_train_batch_size
            per_step_batch = per_gpu_batch * world_size
            total_batch = per_step_batch * grad_acc_steps
            logger.info("\nBatch Size Analysis:")
            logger.info(f"  Per GPU batch size: {per_gpu_batch}")
            logger.info(f"  Per step batch size (across GPUs): {per_step_batch}")
            logger.info(f"  Total batch size (with gradient accumulation): {total_batch}")

        # Calculate final training steps
        self.num_training_steps = int(dataloader_length * num_epochs) // grad_acc_steps // world_size
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)

        # Create scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # Prepare everything with accelerator
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # Initialize metric tracker with accelerator for consistent world size
        self.metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)

        if self.accelerator.is_main_process:
            logger.info("\nTraining Configuration:")
            logger.info(f"  Number of GPUs: {world_size}")
            logger.info(f"  Batch size per GPU: {self.config.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total batch size: {self.config.per_device_train_batch_size * world_size * self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps: {self.num_training_steps}")
            logger.info(f"  Number of warmup steps: {num_warmup_steps}")
            logger.info(f"  Dataset size: {len(train_dataset)}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  Weight decay: {self.config.weight_decay}")
            logger.info(f"  Warmup ratio: {self.config.warmup_ratio}")
            logger.info(f"  Adam betas: ({self.config.adam_beta1}, {self.config.adam_beta2})")
            logger.info(f"  Adam epsilon: {self.config.adam_epsilon}")

    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_final: Whether this is the final checkpoint
        """
        if not self.accelerator.is_main_process:
            return
            
        # For intermediate checkpoints, save in numbered subdirectories
        if not is_final:
            save_dir = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
            os.makedirs(save_dir, exist_ok=True)
        else:
            # For final checkpoint, save directly in output directory
            save_dir = self.config.output_dir
        
        # Get model state dict
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model)
        )
        self.tokenizer.save_pretrained(save_dir)
        
        # Only rotate checkpoints for intermediate saves
        if not is_final:
            checkpoint_dirs = [d for d in os.listdir(self.config.output_dir) 
                             if d.startswith("checkpoint-")]
            if len(checkpoint_dirs) > self.config.max_ckpts:
                oldest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[0]
                oldest_checkpoint_path = os.path.join(self.config.output_dir, oldest_checkpoint)
                if os.path.exists(oldest_checkpoint_path):
                    shutil.rmtree(oldest_checkpoint_path)

    def train(self):
        """Run training loop."""
        # Use the same num_training_steps calculated in setup()
        progress_bar = tqdm(range(self.num_training_steps), disable=not self.accelerator.is_main_process)
        completed_steps = 0
        starting_epoch = 0

        for epoch in range(starting_epoch, self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    
                    # Update metrics
                    self.metric(outputs.logits, batch["labels"], loss)
                    acc, train_loss = self.metric.get_metric()
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    
                    if self.accelerator.is_main_process:
                        progress_bar.set_postfix(
                            loss=f"{train_loss:.3f}",
                            acc=f"{acc:.3f}",
                            lr=f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                        )
                        
                        wandb.log({
                            'loss': train_loss,
                            'accuracy': acc,
                            'learning_rate': self.lr_scheduler.get_last_lr()[0],
                            'epoch': epoch,
                            'step': completed_steps,
                        })

                if completed_steps >= self.num_training_steps:
                    break

            # Save intermediate checkpoint
            self.save_checkpoint(epoch)

        # Save final model directly in output directory
        if self.accelerator.is_main_process:
            logger.info("\nSaving final model...")
            self.save_checkpoint(self.config.num_train_epochs - 1, is_final=True)
            wandb.finish()