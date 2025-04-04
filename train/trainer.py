"""Trainer class for supervised fine-tuning."""

import os
import logging
import datetime
import shutil
import math
import copy
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
        
        # Initialize early stopping variables
        self.early_stopping_counter = 0
        self.best_metric = float('inf') if config.early_stopping_metric == 'loss' else float('-inf')
        self.best_model_state = None
        self.should_stop = False
        
        # Set validation frequency (default to every 0.2 epochs if not specified)
        self.validation_steps = getattr(config, 'validation_steps', None)
        if self.validation_steps is None:
            # Default to validating roughly every 20% of an epoch
            # Will be properly calculated after dataloader is created
            self.validation_frequency = 0.2
        else:
            # Use the specified number of steps directly
            self.validation_frequency = None
        
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

        # Create training dataset
        train_dataset = PreformattedDataset(
            self.config.train_file_path,
            self.tokenizer,
            self.config.block_size,
            self.config.debug,
            split="train"
        )
        
        # Create validation dataset if available
        try:
            val_dataset = PreformattedDataset(
                self.config.train_file_path,
                self.tokenizer,
                self.config.block_size,
                self.config.debug,
                split="validation"
            )
            self.has_validation = True
            logger.info(f"Validation dataset loaded with {len(val_dataset)} examples")
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"No validation dataset found: {e}")
            val_dataset = None
            self.has_validation = False
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn
        )
        
        if self.has_validation:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.per_device_train_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=val_dataset.collate_fn
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
        
        # Calculate validation steps if using frequency-based validation
        if self.validation_frequency is not None and self.has_validation:
            # Calculate steps per epoch (accounting for gradient accumulation and distributed training)
            steps_per_epoch = dataloader_length // grad_acc_steps // world_size
            
            # Adaptive validation frequency based on dataset size
            # For very small datasets, validate less frequently (minimum 2 times per epoch)
            # For large datasets, validate more frequently (up to 10 times per epoch)
            if steps_per_epoch < 20:
                # Small dataset: validate at most 2-3 times per epoch
                target_validations_per_epoch = min(2, max(1, steps_per_epoch // 2))
            elif steps_per_epoch < 100:
                # Medium dataset: validate 3-5 times per epoch
                target_validations_per_epoch = min(5, max(3, steps_per_epoch // 20))
            else:
                # Large dataset: validate up to 10 times per epoch
                target_validations_per_epoch = min(10, max(5, steps_per_epoch // 50))
            
            # Calculate steps between validations
            self.validation_steps = max(1, steps_per_epoch // target_validations_per_epoch)
            
            if self.accelerator.is_main_process:
                actual_frequency = self.validation_steps / steps_per_epoch if steps_per_epoch > 0 else 0
                logger.info(f"  Validation frequency: Every {self.validation_steps} steps")
                logger.info(f"  Approximately {1/actual_frequency:.1f} validations per epoch")

        # Create scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # Prepare everything with accelerator
        if self.has_validation:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
            )
        else:
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
            if self.has_validation:
                logger.info(f"  Validation dataset size: {len(val_dataset)}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  Weight decay: {self.config.weight_decay}")
            logger.info(f"  Warmup ratio: {self.config.warmup_ratio}")
            logger.info(f"  Adam betas: ({self.config.adam_beta1}, {self.config.adam_beta2})")
            logger.info(f"  Adam epsilon: {self.config.adam_epsilon}")

    def validate(self):
        """Evaluate model on validation set."""
        if not self.has_validation:
            return 0.0, 0.0
            
        self.model.eval()
        val_metric = SFTMetric(device=self.accelerator.device, accelerator=self.accelerator)
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                val_metric(outputs.logits, batch["labels"], loss)
        
        # Get validation metrics
        val_acc, val_loss = val_metric.get_metric()
        self.model.train()
        
        return val_loss, val_acc
        
    def check_early_stopping(self, val_loss: float, val_acc: float) -> bool:
        """Check if early stopping criteria are met.
        
        Args:
            val_loss: Validation loss
            val_acc: Validation accuracy
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if not self.config.early_stopping:
            return False
            
        # Determine current metric value based on configuration
        current_metric = val_loss if self.config.early_stopping_metric == 'loss' else val_acc
        
        # For loss, lower is better; for accuracy, higher is better
        is_better = False
        if self.config.early_stopping_metric == 'loss':
            # Check if current loss is better (lower) than best loss by at least the threshold
            is_better = current_metric < (self.best_metric - self.config.early_stopping_threshold)
        else:
            # Check if current accuracy is better (higher) than best accuracy by at least the threshold
            is_better = current_metric > (self.best_metric + self.config.early_stopping_threshold)
        
        if is_better:
            # Reset counter and update best metric
            self.early_stopping_counter = 0
            self.best_metric = current_metric
            
            # Save best model state
            if self.accelerator.is_main_process:
                logger.info(f"New best {self.config.early_stopping_metric}: {current_metric:.4f}")
                self.best_model_state = copy.deepcopy(
                    self.accelerator.get_state_dict(self.model)
                )
            return False
        else:
            # Increment counter
            self.early_stopping_counter += 1
            if self.accelerator.is_main_process:
                logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.config.early_stopping_patience}")
                
            # Check if we should stop
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                if self.accelerator.is_main_process:
                    logger.info(f"Early stopping triggered after {self.early_stopping_counter} epochs without improvement")
                return True
            return False
    
    def save_checkpoint(self, epoch: int, is_final: bool = False, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_final: Whether this is the final checkpoint
            is_best: Whether this is the best model according to early stopping metric
        """
        if not self.accelerator.is_main_process:
            return
            
        # Determine save directory
        if is_best:
            save_dir = os.path.join(self.config.output_dir, "best_model")
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving best model to {save_dir}")
        elif not is_final:
            save_dir = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
            os.makedirs(save_dir, exist_ok=True)
        else:
            # For final checkpoint, save directly in output directory
            save_dir = self.config.output_dir
        
        # Get model state dict
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # If saving best model and we have a stored best state
        if is_best and self.best_model_state is not None:
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.best_model_state
            )
        else:
            unwrapped_model.save_pretrained(
                save_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model)
            )
        
        self.tokenizer.save_pretrained(save_dir)
        
        # Only rotate checkpoints for intermediate saves (not best or final)
        if not is_final and not is_best:
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
        
        # Log validation and early stopping configuration
        if self.accelerator.is_main_process:
            logger.info("\nValidation Configuration:")
            logger.info(f"  Validation Steps: {self.validation_steps}")
            
            if self.config.early_stopping:
                logger.info("\nEarly Stopping Configuration:")
                logger.info(f"  Enabled: {self.config.early_stopping}")
                logger.info(f"  Patience: {self.config.early_stopping_patience}")
                logger.info(f"  Threshold: {self.config.early_stopping_threshold}")
                logger.info(f"  Metric: {self.config.early_stopping_metric}")

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
                            'train_loss': train_loss,
                            'train_accuracy': acc,
                            'learning_rate': self.lr_scheduler.get_last_lr()[0],
                            'epoch': epoch + (step / len(self.train_dataloader)),  # Fractional epoch
                            'step': completed_steps,
                        })
                    
                    # Run validation at specified intervals
                    if self.has_validation and completed_steps % self.validation_steps == 0:
                        val_loss, val_acc = self.validate()
                        if self.accelerator.is_main_process:
                            logger.info(f"Step {completed_steps} Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                            wandb.log({
                                'val_loss': val_loss,
                                'val_accuracy': val_acc,
                                'epoch': epoch + (step / len(self.train_dataloader)),  # Fractional epoch
                                'step': completed_steps,
                            })
                        
                        # Check early stopping criteria
                        if self.config.early_stopping:
                            should_stop = self.check_early_stopping(val_loss, val_acc)
                            
                            # Save best model if this is the best performance so far
                            if self.early_stopping_counter == 0 and self.accelerator.is_main_process:
                                self.save_checkpoint(epoch, is_best=True)
                            
                            # Broadcast early stopping decision to all processes
                            should_stop = self.accelerator.gather(torch.tensor([should_stop], device=self.accelerator.device)).any().item()
                            
                            if should_stop:
                                if self.accelerator.is_main_process:
                                    logger.info("Early stopping triggered, stopping training")
                                break

                if completed_steps >= self.num_training_steps:
                    break

            # Run validation at the end of each epoch (in addition to step-based validation)
            if self.has_validation:
                # Only run end-of-epoch validation if we haven't just done it
                # (which could happen if the last step triggered validation)
                if completed_steps % self.validation_steps != 0:
                    val_loss, val_acc = self.validate()
                    if self.accelerator.is_main_process:
                        logger.info(f"End of epoch {epoch} validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                        wandb.log({
                            'val_loss': val_loss,
                            'val_accuracy': val_acc,
                            'epoch': epoch + 1.0,  # End of epoch
                            'step': completed_steps,
                        })
                    
                    # Check early stopping criteria
                    if self.config.early_stopping:
                        should_stop = self.check_early_stopping(val_loss, val_acc)
                        
                        # Save best model if this is the best performance so far
                        if self.early_stopping_counter == 0 and self.accelerator.is_main_process:
                            self.save_checkpoint(epoch, is_best=True)
                        
                        # Broadcast early stopping decision to all processes
                        should_stop = self.accelerator.gather(torch.tensor([should_stop], device=self.accelerator.device)).any().item()
                        
                        if should_stop:
                            if self.accelerator.is_main_process:
                                logger.info("Early stopping triggered, stopping training")
                            break
            
            # Save intermediate checkpoint
            self.save_checkpoint(epoch)

        # Save final model directly in output directory
        if self.accelerator.is_main_process:
            logger.info("\nSaving final model...")
            
            # If early stopping was enabled and we have a best model state, use that for the final model
            if self.config.early_stopping and self.best_model_state is not None:
                logger.info("Using best model from early stopping for final model")
                self.save_checkpoint(self.config.num_train_epochs - 1, is_final=True, is_best=True)
            else:
                self.save_checkpoint(self.config.num_train_epochs - 1, is_final=True)
                
            wandb.finish()