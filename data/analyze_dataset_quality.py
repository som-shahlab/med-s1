import os
import json
import pandas as pd
import numpy as np
import random
from datasets import load_dataset, Dataset, load_from_disk
import asyncio
from typing import List, Dict, Tuple
from utils.openai_utils import get_model_response
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def load_config() -> Dict:
    """Load configuration from config.json"""
    assert os.getenv("MED_S1_DIR") is not None, "MED_S1_DIR environment variable not set"
    with open(os.path.join(os.getenv("MED_S1_DIR"), "config.json"), "r") as f:
        return json.load(f)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_medical_o1_dataset() -> pd.DataFrame:
    """Load medical-o1-reasoning-SFT dataset"""
    print("\nLoading medical-o1-reasoning-SFT dataset...")
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} examples")
    print("\nSample question:")
    print(f"Question: {df['Question'].iloc[0][:200]}...")
    print(f"Complex_CoT: {df['Complex_CoT'].iloc[0][:200]}...")
    return df

def load_nejmcr_dataset() -> pd.DataFrame:
    """Load NEJM case reports dataset"""
    print("\nLoading NEJM case reports dataset...")
    config = load_config()
    base_path = config["train_datasets"]["nejmcr"]["file_path"]
    
    # Load from arrow file
    dataset = load_from_disk(base_path)
    print(f"Loading dataset from {base_path}")
    
    df = pd.DataFrame(dataset)
    df["Question"] = df["question"] + "\nWhat is the diagnosis of the patient?"
    df["Complex_CoT"] = df["thinking"]
    df["Response"] = df.apply(lambda x: x.get("diagnosis_final", "No diagnosis available"), axis=1)
    df = df[["Question", "Complex_CoT", "Response"]]
    print(f"Loaded {len(df)} examples")
    print("\nSample question:")
    print(f"Question: {df['Question'].iloc[0][:200]}...")
    print(f"Complex_CoT: {df['Complex_CoT'].iloc[0][:200]}...")
    return df

def load_med_s1k_dataset() -> pd.DataFrame:
    """Load med_s1k_curated dataset"""
    print("\nLoading med_s1k_curated dataset...")
    path = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/medqa-nejmcr-1k-random-nejmcr-qa-reason-qwen-tuned_20250410_132718/med_s1k_curated.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} examples")
    print("\nSample question:")
    print(f"Question: {df['Question'].iloc[0][:200]}...")
    print(f"Complex_CoT: {df['Complex_CoT'].iloc[0][:200]}...")
    return df

def filter_diagnosis_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Filter questions containing 'diagnosis' but not 'A.'"""
    print("\nFiltering diagnosis questions...")
    mask = df["Question"].str.contains("diagnosis", case=False) & ~df["Question"].str.contains("A\.")
    filtered_df = df[mask]
    print(f"Filtered from {len(df)} to {len(filtered_df)} questions")
    return filtered_df

def sample_dataset(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Sample n rows from dataset"""
    print(f"\nSampling {n} examples...")
    if len(df) <= n:
        print(f"Dataset has only {len(df)} examples, using all")
        return df
    return df.sample(n=n, random_state=seed)

def compute_length(text: str) -> int:
    """Compute word count of text"""
    return len(text.split())

async def process_sentence_batch(question: str, sentences: List[str], start_idx: int) -> Tuple[int, List[Dict]]:
    """Process a batch of sentences for relevance"""
    labeled_sentences = [f"{chr(65+i)}. {s}" for i, s in enumerate(sentences)]
    
    prompt = f"Question: {question}\n\nBelow are sentences from a reasoning trace. For each sentence, determine if it is relevant to answering the question.\nList only the letters of relevant sentences, separated by commas.\n\n{chr(10).join(labeled_sentences)}\n\nRelevant sentences (letters only, comma-separated):"
    
    response = await get_model_response(prompt, model="gemini-2.0-flash")
    relevant_letters = re.findall(r'[A-Z]', response) if response else []
    
    return len(relevant_letters), [{
        "batch_start": start_idx,
        "sentences": sentences,
        "prompt": prompt,
        "response": response,
        "relevant_count": len(relevant_letters)
    }]

async def compute_relevance(question: str, reasoning: str) -> Tuple[float, List[Dict]]:
    """Compute fraction of relevant sentences in reasoning"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', reasoning) if s.strip()]
    batch_size = 40
    total_relevant = 0
    intermediate_results = []
    
    tasks = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        tasks.append(process_sentence_batch(question, batch, i))
    
    results = await asyncio.gather(*tasks)
    
    for relevant_count, batch_results in results:
        total_relevant += relevant_count
        intermediate_results.extend(batch_results)
    
    return total_relevant / len(sentences) if sentences else 0, intermediate_results

async def compute_exploration(question: str, reasoning: str) -> Tuple[int, Dict]:
    """Compute number of answer choices explored"""
    prompt = f"Question: {question}\n\nReasoning trace: {reasoning}\n\nList each unique answer choice or diagnosis that was explored or considered in the reasoning trace.\nPut each one on a separate line. Include only explicitly mentioned options."
    
    response = await get_model_response(prompt, model="gemini-2.0-flash")
    count = len([line for line in response.split('\n') if line.strip()]) if response else 0
    
    return count, {
        "prompt": prompt,
        "response": response,
        "count": count
    }

async def process_example(example: Dict) -> Dict:
    """Process a single example computing all metrics"""
    length = compute_length(example["Complex_CoT"])
    relevance, relevance_results = await compute_relevance(example["Question"], example["Complex_CoT"])
    exploration, exploration_results = await compute_exploration(example["Question"], example["Complex_CoT"])
    
    return {
        "question": example["Question"],
        "reasoning": example["Complex_CoT"],
        "metrics": {
            "length": length,
            "relevance": relevance,
            "exploration": exploration
        },
        "intermediate_results": {
            "relevance": relevance_results,
            "exploration": exploration_results
        }
    }

async def analyze_dataset(df: pd.DataFrame, name: str) -> Dict:
    """Analyze dataset quality metrics"""
    print(f"\nAnalyzing {name} dataset ({len(df)} examples)...")
    
    results = {
        "dataset_name": name,
        "sample_size": len(df),
        "examples": []
    }
    
    # Process examples in parallel
    examples = df.to_dict('records')
    tasks = [process_example(example) for example in examples]
    
    # Use tqdm to show progress
    examples_results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing examples"):
        examples_results.append(await task)
    
    results["examples"] = examples_results
    
    # Compute aggregate metrics
    metrics = {
        "length": {"values": [], "mean": 0, "variance": 0},
        "relevance": {"values": [], "mean": 0, "variance": 0},
        "exploration": {"values": [], "mean": 0, "variance": 0}
    }
    
    for example in examples_results:
        for metric, value in example["metrics"].items():
            metrics[metric]["values"].append(value)
    
    for metric, data in metrics.items():
        values = data["values"]
        data["mean"] = np.mean(values)
        data["variance"] = np.var(values)
        print(f"{metric.capitalize()}: mean = {data['mean']:.2f}, variance = {data['variance']:.2f}")
    
    results["metrics"] = metrics
    return results

async def main():
    # Set random seed
    set_random_seeds(42)
    
    # Load datasets
    medical_o1_df = load_medical_o1_dataset()
    nejmcr_df = load_nejmcr_dataset()
    med_s1k_df = load_med_s1k_dataset()
    
    # Filter medical_o1 dataset
    medical_o1_df = filter_diagnosis_questions(medical_o1_df)
    
    # Sample datasets
    medical_o1_sample = sample_dataset(medical_o1_df)
    nejmcr_sample = sample_dataset(nejmcr_df)
    med_s1k_sample = sample_dataset(med_s1k_df)
    
    # Analyze datasets
    results = {
        "medical_o1": await analyze_dataset(medical_o1_sample, "medical-o1-reasoning-SFT"),
        "nejmcr": await analyze_dataset(nejmcr_sample, "NEJM Case Reports"),
        "med_s1k": await analyze_dataset(med_s1k_sample, "MedQA NEJMCR 1k")
    }
    
    # Save results
    output_path = "dataset_quality_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())