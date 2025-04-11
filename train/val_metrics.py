"""Metrics for validation during training."""

import torch
import torch.distributed as dist
import logging
import re
import difflib
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class ValMetric:
    """Metric tracker for validation.
    
    This class tracks two types of metrics:
    1. Language modeling metrics (like training):
       - Loss on formatted text
       - Token-level accuracy
       
    2. Multiple choice metrics (like evaluation):
       - Answer accuracy per source
       - Overall answer accuracy
    """
    
    def __init__(self, device, tokenizer, accelerator=None):
        """Initialize metric tracker.
        
        Args:
            device: Device to store tensors on
            tokenizer: Tokenizer for decoding outputs
            accelerator: Accelerator instance for distributed training info
        """
        self.tokenizer = tokenizer
        # Language modeling metrics
        self.n_step = 0
        self.token_right = torch.Tensor([0]).to(device=device)
        self.token_total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        
        # Multiple choice metrics
        self.source_metrics = {}  # Tracks per-source metrics
        self.device = device
        
        # Get world size from accelerator if provided, otherwise fallback to dist
        self.world_size = accelerator.num_processes if accelerator else dist.get_world_size()
        
        # Log initialization
        if accelerator and accelerator.is_main_process:
            logger.info(f"ValMetric initialized:")
            logger.info(f"  Device: {device}")
            logger.info(f"  World size: {self.world_size}")
    
    def str_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using sequence matcher."""
        seq = difflib.SequenceMatcher(None, str1, str2)
        return seq.ratio()
    
    def find_most_similar_index(self, str_list: List[str], target_str: str) -> int:
        """Find index of most similar string in list."""
        highest_similarity = 0
        most_similar_index = None
        
        for i, str in enumerate(str_list):
            similarity = self.str_similarity(str, target_str)
            if similarity >= highest_similarity:
                most_similar_index = i
                highest_similarity = similarity
    
        return most_similar_index
    
    def match_choice(self, text: str, options: Dict[str, str]) -> Tuple[List[str], int]:
        """Extract model's answer choice from output text.
        
        Uses same matching logic as evaluation pipeline:
        1. Strict prompt matching
        2. Non-strict matching
        3. Option text matching
        4. Fallback to most similar text
        
        Returns:
            Tuple of:
            - List containing [first_match, last_match]
            - Match type (1=exact, 2=option text, 3=similarity)
        """
        # Split on special tokens if present
        if '<|start_header_id|>answer<|end_header_id|>' in text:
            text = text.split('<|start_header_id|>answer<|end_header_id|>')[-1]
        if 'Answer:' in text:
            text = text.split('Answer:')[-1]
        if '## Final Response\n\n' in text:
            text = text.split('## Final Response\n\n')[-1]
        if '</think>' in text:
            text = text.split('</think>')[-1]
        
        # Try strict prompt matching
        matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
        if matches:
            ans_first = matches[0].group(2)
            ans_last = matches[-1].group(2)
            return [ans_first, ans_last], 1
    
        # Try non-strict matching
        match_options = 'ABCDEFGHIJKLMN'[:len(options)]
        matches = list(re.finditer(
            r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)",
            text, re.S
        ))
        if matches:
            ans_first = matches[0].group(2)
            ans_last = matches[-1].group(2)
            return [ans_first, ans_last], 1
    
        # Try matching option text
        text = text.lower()
        opsindex = [(opt, text.rindex(options[opt].lower()))
                    for opt in options if options[opt].lower() in text]
        if opsindex:
            ans_last = sorted(opsindex, key=lambda x:x[1], reverse=True)[0][0]
            opsindex = [(opt, text.index(options[opt].lower()))
                        for opt in options if options[opt].lower() in text]
            ans_first = sorted(opsindex, key=lambda x:x[1], reverse=True)[0][0]
            return [ans_first, ans_last], 2
        
        # Fall back to most similar text
        oplabels = [x for x in options]
        opans = [options[x].lower() for x in options]
        ansindex = self.find_most_similar_index(opans, text.lower())
        return [oplabels[ansindex], oplabels[ansindex]], 3
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor, 
               metadata: List[Dict]):
        """Update metrics with new batch.
        
        Args:
            logits: Model output logits
            labels: Ground truth labels
            loss: Loss value
            metadata: List of dicts containing:
                - source: Source dataset
                - options: Answer options
                - answer_idx: Correct answer
                - id: Sample ID
        """
        self.n_step += 1
        with torch.no_grad():
            # Update language modeling metrics
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.token_right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.token_total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()
            
            # Generate answers for multiple choice accuracy
            generated = []
            for i in range(logits.size(0)):
                # Get tokens up to first padding
                tokens = logits[i].argmax(dim=-1)
                text = self.tokenizer.decode(tokens)
                generated.append(text)
            
            # Update multiple choice metrics
            for text, meta in zip(generated, metadata):
                source = meta['source']
                if source not in self.source_metrics:
                    self.source_metrics[source] = {
                        'correct': torch.Tensor([0]).to(self.device),
                        'total': torch.Tensor([0]).to(self.device)
                    }
                
                # Get model's answer and check if correct
                ans, _ = self.match_choice(text, meta['options'])
                # Use last matched answer (like evaluation)
                correct = int(ans[-1].lower() == meta['answer_idx'].lower())
                
                # Update source metrics
                self.source_metrics[source]['correct'] += correct
                self.source_metrics[source]['total'] += 1
    
    def get_metric(self, reset: bool = True) -> Tuple[Dict, float]:
        """Get current metrics, optionally resetting counters.
        
        Args:
            reset: Whether to reset counters after getting metrics
            
        Returns:
            Tuple of:
            - Dict containing per-source metrics
            - Overall validation loss
        """
        # Synchronize language modeling metrics
        dist.all_reduce(self.token_right, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.token_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=dist.ReduceOp.SUM)
        
        # Calculate language modeling metrics
        token_acc = (self.token_right / self.token_total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)
        
        # Synchronize multiple choice metrics
        metrics = {}
        total_correct = torch.Tensor([0]).to(self.device)
        total_samples = torch.Tensor([0]).to(self.device)
        
        for source, source_metrics in self.source_metrics.items():
            # Synchronize source metrics
            dist.all_reduce(source_metrics['correct'], op=dist.ReduceOp.SUM)
            dist.all_reduce(source_metrics['total'], op=dist.ReduceOp.SUM)
            
            # Calculate source accuracy
            correct = source_metrics['correct'].item()
            total = source_metrics['total'].item()
            accuracy = correct / total if total > 0 else 0.0
            
            metrics[source] = {
                'accuracy': accuracy,
                'total_samples': total
            }
            
            # Update overall totals
            total_correct += source_metrics['correct']
            total_samples += source_metrics['total']
        
        # Calculate overall accuracy
        overall_accuracy = (total_correct / total_samples).item() if total_samples.item() > 0 else 0.0
        metrics['overall'] = {
            'accuracy': overall_accuracy,
            'total_samples': total_samples.item(),
            'token_accuracy': token_acc,
            'loss': loss
        }
        
        # Log metrics
        logger.info("\nValidation Metrics:")
        logger.info(f"  Language Modeling:")
        logger.info(f"    Loss: {loss:.4f}")
        logger.info(f"    Token Accuracy: {token_acc:.4f}")
        logger.info(f"  Multiple Choice:")
        for source, source_metrics in metrics.items():
            if source != 'overall':
                logger.info(f"    {source}:")
                logger.info(f"      Accuracy: {source_metrics['accuracy']:.4f}")
                logger.info(f"      Samples: {source_metrics['total_samples']}")
        logger.info(f"  Overall:")
        logger.info(f"    Accuracy: {overall_accuracy:.4f}")
        logger.info(f"    Total Samples: {total_samples.item()}")
        
        # Reset counters if requested
        if reset:
            self.n_step = 0
            self.token_right.fill_(0)
            self.token_total.fill_(0)
            self.total_loss.fill_(0)
            self.source_metrics = {}
        
        return metrics, loss