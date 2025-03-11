"""Metrics for tracking training progress."""

import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

class SFTMetric:
    """Metric tracker for supervised fine-tuning.
    
    This class tracks accuracy and loss metrics across all GPUs in distributed training.
    It uses dist.all_reduce to aggregate metrics across processes, exactly matching
    HuatuoGPT's implementation.
    """
    
    def __init__(self, device, accelerator=None):
        """Initialize metric tracker.
        
        Args:
            device: Device to store tensors on
            accelerator: Accelerator instance for distributed training info
        """
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        
        # Get world size from accelerator if provided, otherwise fallback to dist
        self.world_size = accelerator.num_processes if accelerator else dist.get_world_size()
        
        # Log initialization
        if accelerator and accelerator.is_main_process:
            logger.info(f"SFTMetric initialized:")
            logger.info(f"  Device: {device}")
            logger.info(f"  World size: {self.world_size}")
            if accelerator:
                logger.info(f"  Using world size from accelerator")
            else:
                logger.info(f"  Using world size from torch.distributed")

    def __call__(self, logits, labels, loss):
        """Update metrics with new batch.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss: Loss value
        """
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        """Update metrics with new batch.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss: Loss value
        """
        self.n_step += 1
        with torch.no_grad():
            # Calculate accuracy
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            
            # Track loss
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        """Get current metrics, optionally resetting counters.
        
        Args:
            reset: Whether to reset counters after getting metrics
            
        Returns:
            Tuple of (accuracy, loss)
        """
        # Synchronize metrics across processes
        dist.all_reduce(self.right, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=dist.ReduceOp.SUM)

        # Calculate metrics
        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)
        
        # Log raw values for debugging
        logger.debug(f"Metric calculation:")
        logger.debug(f"  Raw values:")
        logger.debug(f"    Right: {self.right.item()}")
        logger.debug(f"    Total: {self.total.item()}")
        logger.debug(f"    Loss: {self.total_loss.item()}")
        logger.debug(f"    N steps: {self.n_step}")
        logger.debug(f"    World size: {self.world_size}")
        logger.debug(f"  Calculated metrics:")
        logger.debug(f"    Accuracy: {acc}")
        logger.debug(f"    Loss: {loss}")

        # Reset counters if requested
        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
            
        return acc, loss