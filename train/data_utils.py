"""Dataset utilities for medical dialogue model training."""

import logging
import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

logger = logging.getLogger(__name__)

class PreformattedDataset(Dataset):
    """Dataset class for handling pre-formatted medical dialogue data.
    
    This dataset expects data in the format:
    formatted_dataset_huatuo['train'][idx]['text'] = 
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>"
    """
    
    def __init__(self, data_path: str, tokenizer, block_size: int, debug: bool = False, split: str = "train"):
        """Initialize dataset.
        
        Args:
            data_path: Path to the HuggingFace dataset
            tokenizer: Tokenizer for encoding text
            block_size: Maximum sequence length
            debug: If True, use reduced dataset size
            split: Dataset split to use ("train" or "validation")
        """
        # Load dataset
        logger.info(f"Loading dataset from: {data_path}")
        
        # Check if data_path is a directory with train/validation subdirectories
        train_dir = os.path.join(data_path, 'train')
        validation_dir = os.path.join(data_path, 'validation')
        test_dir = os.path.join(data_path, 'test')
        
        if os.path.exists(train_dir) and (os.path.exists(validation_dir) or os.path.exists(test_dir)):
            # New format with separate directories
            if split == "train":
                logger.info(f"Loading training split from {train_dir}")
                self.dataset = load_from_disk(train_dir)
            elif split == "validation" and os.path.exists(validation_dir):
                logger.info(f"Loading validation split from {validation_dir}")
                self.dataset = load_from_disk(validation_dir)
            elif split == "validation" and os.path.exists(test_dir):
                logger.info(f"Loading validation split from test directory {test_dir}")
                self.dataset = load_from_disk(test_dir)
            else:
                raise ValueError(f"Split {split} not found in {data_path}")
        else:
            # Old format with dataset_dict.json
            dataset = load_from_disk(data_path)
            
            # Check available splits
            if 'train' in dataset and split == 'train':
                self.dataset = dataset['train']
            elif 'validation' in dataset and split == 'validation':
                self.dataset = dataset['validation']
            elif 'test' in dataset and split == 'validation':
                # Use test split for validation if validation not available
                self.dataset = dataset['test']
            else:
                raise ValueError(f"Split {split} not found in dataset")
        
        # Log dataset size
        logger.info(f"Loaded dataset with {len(self.dataset)} samples")
        
        # Handle debug mode
        if debug:
            min_samples = 32  # At least 8 samples per GPU
            logger.warning(f"Debug mode: Using {min_samples} samples")
            self.dataset = self.dataset.select(range(min(min_samples, len(self.dataset))))
            logger.warning(f"Debug dataset size: {len(self.dataset)}")
            logger.warning(f"First example: {self.dataset[0]}")
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.debug = 0
        
    def __getitem__(self, idx):
        """Get a single example."""
        # Get pre-formatted text
        text = self.dataset[idx]['text']
        
        # Verify format based on model markers
        if '<|im_start|>' in text:
            # Qwen format
            assert text.startswith('<|im_start|>'), f"Qwen text doesn't start with im_start: {text[:50]}"
            assert '<|im_end|>' in text, f"Qwen text doesn't contain im_end: {text[-50:]}"
        else:
            # Default format (Huatuo/LLaMA)
            assert text.startswith('<|begin_of_text|>'), f"Text doesn't start with begin_of_text: {text[:50]}"
            assert '<|eot_id|>' in text, f"Text doesn't contain eot_id: {text[-50:]}"
        
        return text
        
    def __len__(self):
        """Get dataset size."""
        return len(self.dataset)
    
    def get_query_and_response(self, text):
        """Split text into query and response parts."""
        # Find the assistant section which contains the response
        if '<|im_start|>' in text:
            # Qwen format
            assistant_start = text.find('<|im_start|>assistant')
            assert assistant_start != -1, "No assistant section found in Qwen format"
        else:
            # Default format (Huatuo/LLaMA)
            assistant_start = text.find('<|start_header_id|>assistant<|end_header_id|>')
            assert assistant_start != -1, "No assistant section found"
        
        # Query is everything before assistant's response
        query = text[:assistant_start]
        # Response is everything after, including the assistant header
        response = text[assistant_start:]
        
        # Verify format based on model markers
        if '<|im_start|>' in text:
            # Qwen format
            assert query.startswith('<|im_start|>'), f"Qwen query doesn't start with im_start: {query[:50]}"
            assert '<|im_end|>' in query, f"Qwen query doesn't contain im_end: {query[-50:]}"
            assert '<|im_end|>' in response, f"Qwen response doesn't contain im_end: {response[-50:]}"
        else:
            # Default format (Huatuo/LLaMA)
            assert query.startswith('<|begin_of_text|>'), f"Query doesn't start with begin_of_text: {query[:50]}"
            assert '<|eot_id|>' in query, f"Query doesn't contain eot_id: {query[-50:]}"
            assert '<|eot_id|>' in response, f"Response doesn't contain eot_id: {response[-50:]}"
        
        return query, response
    
    def collate_fn(self, batch):
        """Custom collation function for creating batches."""
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
        
        # Log first few examples in debug mode
        if self.debug < 3:
            logger.debug(f"Input text: {self.tokenizer.decode(input_ids_list[-1])}")
            logger.debug(f"Labels text: {self.tokenizer.decode([0 if x == -100 else x for x in labels_list[-1]])}")
            self.debug += 1
        
        return {
            "input_ids": torch.LongTensor(input_ids_list),
            "labels": torch.LongTensor(labels_list)
        }