"""Dataset class for validation during training."""

import json
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ValDataset(Dataset):
    """Dataset for validation during training.
    
    This dataset serves two purposes:
    1. Compute validation loss using same format as training data
    2. Compute validation accuracy using same metrics as evaluation
    
    For loss computation, it formats data like training:
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>"
    
    For accuracy computation, it tracks:
    - Multiple choice options
    - Correct answer index
    - Source dataset
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        block_size: int,
        debug: bool = False
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to validation data JSON file
            tokenizer: Tokenizer for encoding text
            block_size: Maximum sequence length
            debug: If True, use reduced dataset size
        """
        # Load dataset
        logger.info(f"Loading validation data from: {data_path}")
        with open(data_path, 'r') as f:
            self.raw_data = json.load(f)
        
        # Log dataset size
        logger.info(f"Loaded dataset with {len(self.raw_data)} samples")
        
        # Handle debug mode
        if debug:
            min_samples = 32
            logger.warning(f"Debug mode: Using {min_samples} samples")
            self.raw_data = self.raw_data[:min(min_samples, len(self.raw_data))]
            logger.warning(f"Debug dataset size: {len(self.raw_data)}")
            logger.warning(f"First example: {self.raw_data[0]}")
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.debug = 0
        
        # Store raw data for processing
        self.raw_data = self.raw_data
    
    def format_example(self, item: Dict) -> Tuple[str, str]:
        """Format example for validation.
        
        Returns:
            Tuple of (input_text, target_text)
            
        Note: For Qwen format, model will generate:
        <|im_start|>think\n[reasoning]\n<|im_start|>answer\n[answer]
        
        We only compute loss on the answer part.
        """
        # Format input (just question and options)
        input_text = f"<|im_start|>user\n{item['query']}\n\nOptions:\n"
        options = item['options']
        for opt in sorted(options.keys()):
            input_text += f"{opt}. {options[opt]}\n"
        input_text += "<|im_end|>"
        
        # Format target (just the answer part)
        target_text = f"<|im_start|>answer\n{item['answer']}<|im_end|>"
        
        return input_text, target_text
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single example."""
        item = self.raw_data[idx]
        
        # Format example
        input_text, target_text = self.format_example(item)
        
        # Encode input
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        
        # Add assistant start and think tokens
        assistant_start = "<|im_start|>assistant\n"
        think_start = "<|im_start|>think\n"
        
        # Encode full sequence
        full_text = input_text + assistant_start + think_start
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # Create labels array initialized to -100
        labels = [-100] * len(input_ids)
        
        # Encode target (answer part)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        
        # Only compute loss on answer part
        labels[-len(target_ids):] = target_ids
        
        # Truncate if needed
        if len(input_ids) > self.block_size:
            input_ids = input_ids[-self.block_size:]
            labels = labels[-self.block_size:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'source': item['source'],
            'options': item['options'],
            'answer_idx': item['answer_idx'],
            'id': item['id']
        }
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.formatted_data)
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collation function for creating batches."""
        # Prepare tensors
        input_ids_list = []
        labels_list = []
        metadata = []
        
        for item in batch:
            input_ids_list.append(item['input_ids'])
            labels_list.append(item['labels'])
            metadata.append({
                'source': item['source'],
                'options': item['options'],
                'answer_idx': item['answer_idx'],
                'id': item['id']
            })
        
        # Get max length for padding
        max_length = max(len(ids) for ids in input_ids_list)
        max_length = min(max_length, self.block_size)
        
        # Pad sequences
        input_ids_list = [
            ids + [self.tokenizer.eos_token_id] * (max_length - len(ids))
            for ids in input_ids_list
        ]
        labels_list = [
            labs + [-100] * (max_length - len(labs))
            for labs in labels_list
        ]
        
        # Log first example in debug mode
        if self.debug == 0:
            logger.info("\nFirst validation example:")
            logger.info("Input:")
            logger.info(self.tokenizer.decode(input_ids_list[0]))
            logger.info("\nLabels (non-masked):")
            labels = labels_list[0]
            answer_text = self.tokenizer.decode([x for x in labels if x != -100])
            logger.info(f"Answer text: {answer_text}")
            logger.info("\nMetadata:")
            logger.info(json.dumps(metadata[0], indent=2))
            self.debug += 1
        
        return {
            'input_ids': torch.LongTensor(input_ids_list),
            'labels': torch.LongTensor(labels_list),
            'metadata': metadata
        }