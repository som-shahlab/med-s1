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

def estimate_model_costs(n_examples: int, avg_lengths: Dict[str, float], batch_size: int = 32, model_name: str = "gpt4o-mini") -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Estimate API costs with batching for different models"""
    # Load model pricing from config
    config = load_config()
    model_config = config["models"][model_name]
    pricing = model_config["pricing"]
    
    # Calculate number of batches
    n_batches = (n_examples + batch_size - 1) // batch_size
    
    # Token counts for verify_answer
    VERIFY_TEMPLATE_TOKENS = 150  # Template text
    verify_input_tokens = n_batches * (
        batch_size * (
            avg_lengths["question"] +  # Question
            avg_lengths["response"] +  # Model answer
            avg_lengths["response"]    # Correct answer
        ) +
        VERIFY_TEMPLATE_TOKENS     # Template
    )
    verify_output_tokens = n_batches * batch_size * 50  # Yes/No with explanation
    
    # Token counts for label_specialty
    SPECIALTY_LIST_TOKENS = 500    # List of specialties
    LABEL_TEMPLATE_TOKENS = 200    # Template text
    label_input_tokens = n_batches * (
        batch_size * avg_lengths["question"] +  # Questions
        SPECIALTY_LIST_TOKENS +            # Specialties list
        LABEL_TEMPLATE_TOKENS              # Template
    )
    label_output_tokens = n_batches * batch_size * 100  # Specialty with explanation
    
    # Calculate costs
    input_costs = {
        "verify": (verify_input_tokens / 1_000_000) * pricing["input"],
        "label": (label_input_tokens / 1_000_000) * pricing["input"]
    }
    output_costs = {
        "verify": (verify_output_tokens / 1_000_000) * pricing["output"],
        "label": (label_output_tokens / 1_000_000) * pricing["output"]
    }
    
    # Calculate time based on rate limits
    if "rpm" in pricing:
        total_requests = n_batches * 2  # verify + label
        minutes_needed = total_requests / pricing["rpm"]
        time_estimate = minutes_needed * 60  # convert to seconds
    else:
        time_estimate = 0
    
    return input_costs, output_costs, time_estimate

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
    
    # Calculate percentiles for CoT lengths
    percentiles = [50, 75, 90, 95, 99]
    cot_percentiles = np.percentile(cot_lengths, percentiles)
    
    logging.info("\nToken length statistics:")
    logging.info(f"Questions: mean={np.mean(q_lengths):.1f}, median={np.median(q_lengths):.1f}, max={max(q_lengths)}")
    logging.info(f"\nReasoning trace lengths:")
    logging.info(f"- Mean: {np.mean(cot_lengths):.1f}")
    logging.info(f"- Median: {np.median(cot_lengths):.1f}")
    logging.info(f"- Max: {max(cot_lengths)}")
    for p, v in zip(percentiles, cot_percentiles):
        logging.info(f"- {p}th percentile: {v:.1f}")
    logging.info(f"\nResponses: mean={np.mean(r_lengths):.1f}, median={np.median(r_lengths):.1f}, max={max(r_lengths)}")
    
    # Print distribution of examples above thresholds
    thresholds = [1000, 2000, 3000, 4000, 5000]
    for t in thresholds:
        n_above = sum(1 for l in cot_lengths if l >= t)
        pct_above = (n_above / len(cot_lengths)) * 100
        logging.info(f"Examples with CoT length >= {t}: {n_above} ({pct_above:.1f}%)")
    
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
    
    # Estimate costs with batching for both models
    batch_size = load_config()["curation"]["batch_size"]
    
    # Calculate average lengths for cost estimation
    avg_lengths = {
        "question": np.mean(q_lengths),
        "response": np.mean(r_lengths)
    }
    
    # Get costs for both models
    gpt_inputs, gpt_outputs, gpt_time = estimate_model_costs(len(dataset), avg_lengths, batch_size, "gpt4o-mini")
    gem_inputs, gem_outputs, gem_time = estimate_model_costs(len(dataset), avg_lengths, batch_size, "gemini-2.0-flash")
    
    # Calculate totals
    gpt_total = sum(gpt_inputs.values()) + sum(gpt_outputs.values())
    gem_total = sum(gem_inputs.values()) + sum(gem_outputs.values())
    
    logging.info(f"\nEstimated costs for {len(dataset)} examples (batch_size={batch_size}):")
    
    logging.info("\nGPT-4-mini breakdown:")
    logging.info(f"Verify answers: ${gpt_inputs['verify']:.2f} input + ${gpt_outputs['verify']:.2f} output")
    logging.info(f"Label specialties: ${gpt_inputs['label']:.2f} input + ${gpt_outputs['label']:.2f} output")
    logging.info(f"Total: ${gpt_total:.2f}")
    logging.info(f"Estimated time: {gpt_time:.1f}s ({gpt_time/60:.1f}m)")
    
    logging.info("\nGemini 2.0 Flash breakdown:")
    logging.info(f"Verify answers: ${gem_inputs['verify']:.2f} input + ${gem_outputs['verify']:.2f} output")
    logging.info(f"Label specialties: ${gem_inputs['label']:.2f} input + ${gem_outputs['label']:.2f} output")
    logging.info(f"Total: ${gem_total:.2f}")
    logging.info(f"Estimated time: {gem_time:.1f}s ({gem_time/60:.1f}m)")
    
    # Cost comparison
    savings = gpt_total - gem_total
    savings_pct = (savings / gpt_total) * 100
    logging.info(f"\nSavings using Gemini: ${savings:.2f} ({savings_pct:.1f}%)")

if __name__ == "__main__":
    main()