import os
import json
import pandas as pd
from typing import Dict, Tuple, List, Set
import logging
from collections import Counter
import numpy as np
from tqdm import tqdm
import re
from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

# Common English stop words to exclude
STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
    'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
    'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
    'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
    'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
    'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
    'give', 'day', 'most', 'us', 'is', 'am', 'are', 'was', 'were',
    'been', 'being', 'has', 'had', 'does', 'did', 'doing'
}

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text"""
    return len(tokenizer(text).input_ids)

def analyze_cot_words(texts: list) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Get word frequencies and document frequencies in reasoning traces"""
    # Track both raw frequency and document frequency
    word_freqs = Counter()
    doc_freqs = Counter()
    total_docs = len([texts for text in texts if len(text) >= 2_500])  # Only consider long reasoning traces    
    
    for text in texts:
        if len(text) < 2_500:  # Skip very short reasoning traces
            continue

        # Get unique words in this text
        words = set()
        for word in re.findall(r'\b\w+\b', text.lower()):
            if word.isalpha() and word not in STOP_WORDS and len(word) > 2:  # Only alphabetic words, no numbers or stop words
                word_freqs[word] += 1
                words.add(word)
        
        # Update document frequencies
        for word in words:
            doc_freqs[word] += 1
    
    # Convert document frequencies to percentages
    doc_percentages = {word: (count/total_docs)*100 
                      for word, count in doc_freqs.most_common()}
    
    return word_freqs, doc_percentages

def estimate_gpt4_mini_costs(n_examples: int, batch_size: int = 32) -> Tuple[float, float]:
    """Estimate GPT-4-mini API costs with batching"""
    # Costs per 1M tokens
    INPUT_COST = 0.15  # $0.15 per 1M tokens
    OUTPUT_COST = 0.60  # $0.60 per 1M tokens
    
    # Average token counts (based on typical medical QA)
    AVG_QUESTION_TOKENS = 200
    AVG_ANSWER_TOKENS = 200
    AVG_PROMPT_CONTEXT = 400  # For specialty labeling/verification prompts
    AVG_OUTPUT_TOKENS = 50
    
    # We make 2 API calls per batch:
    # 1. Verify base model answers (batch_size questions at once)
    # 2. Label specialties (batch_size questions at once)
    n_batches = (n_examples + batch_size - 1) // batch_size
    
    # For each batch:
    # Input: batch_size * (question + answer) + prompt_context
    # Output: batch_size * output_tokens
    total_input_tokens = n_batches * (
        batch_size * (AVG_QUESTION_TOKENS + AVG_ANSWER_TOKENS) + AVG_PROMPT_CONTEXT
    )
    total_output_tokens = n_batches * (batch_size * AVG_OUTPUT_TOKENS)
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST
    
    return input_cost, output_cost

def format_word_freq(word: str, freq: int, doc_pct: float) -> str:
    """Format word frequency with document percentage"""
    return f"{word} ({freq}, {doc_pct:.1f}%)"

def main():
    # Load dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    logging.info(f"Dataset size: {len(dataset)} examples")
    
    # Load tokenizer
    model_config = load_config()["models"][load_config()["model_choices"]["base"]]
    tokenizer = AutoTokenizer.from_pretrained(model_config["hf_path"])
    
    # Analyze token lengths
    questions = dataset['Question']
    cots = dataset['Complex_CoT']
    responses = dataset['Response']
    
    q_lengths = []
    cot_lengths = []
    r_lengths = []
    
    for q, c, r in tqdm(zip(questions, cots, responses), total=len(dataset)):
        q_lengths.append(count_tokens(q, tokenizer))
        cot_lengths.append(count_tokens(c, tokenizer))
        r_lengths.append(count_tokens(r, tokenizer))
    
    logging.info("\nToken length statistics:")
    logging.info(f"Questions: mean={np.mean(q_lengths):.1f}, median={np.median(q_lengths):.1f}, max={max(q_lengths)}")
    logging.info(f"Reasoning: mean={np.mean(cot_lengths):.1f}, median={np.median(cot_lengths):.1f}, max={max(cot_lengths)}")
    logging.info(f"Responses: mean={np.mean(r_lengths):.1f}, median={np.median(r_lengths):.1f}, max={max(r_lengths)}")
    
    # Analyze word frequencies in reasoning traces
    word_freqs, doc_percentages = analyze_cot_words(cots)
    
    logging.info("\nMost common words in reasoning traces (word (frequency, % of traces)):")
    # Format as columns
    n_columns = 2
    n_words = 50
    words_per_column = (n_words + n_columns - 1) // n_columns
    
    # Get top words by document frequency
    top_words = sorted(doc_percentages.items(), key=lambda x: x[1], reverse=True)[:n_words]
    columns = []
    
    for i in range(n_columns):
        start = i * words_per_column
        end = min(start + words_per_column, n_words)
        column = [format_word_freq(w, word_freqs[w], p) 
                 for w, p in top_words[start:end]]
        columns.append(column)
    
    # Print columns side by side
    for row in zip(*columns):
        logging.info("  ".join(word.ljust(40) for word in row))
    
    # Estimate costs with batching
    batch_size = load_config()["curation"]["batch_size"]
    input_cost, output_cost = estimate_gpt4_mini_costs(len(dataset), batch_size)
    total_cost = input_cost + output_cost
    logging.info(f"\nEstimated GPT-4-mini costs (with batch_size={batch_size}):")
    logging.info(f"Input cost: ${input_cost:.2f}")
    logging.info(f"Output cost: ${output_cost:.2f}")
    logging.info(f"Total cost: ${total_cost:.2f}")

if __name__ == "__main__":
    main()