import os
import json
import pandas as pd
import asyncio
from datasets import Dataset, load_from_disk
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.openai_utils import get_model_response
from functools import lru_cache

# Get MED_S1_DIR from environment with default
MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

def setup_logging():
    """Configure logging with consistent format"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

@lru_cache(maxsize=1)
def load_config() -> Dict:
    """Load configuration from config.json"""
    with open(os.path.join(MED_S1_DIR, "config.json"), "r") as f:
        return json.load(f)

def load_nejmcr_dataset() -> pd.DataFrame:
    """Load the NEJMCR dataset"""
    config = load_config()
    
    # Get dataset config
    dataset_config = config["train_datasets"]["nejmcr"]
    file_path = dataset_config["file_path"]
    
    logging.info(f"Loading NEJMCR dataset from {file_path}")
    try:
        dataset = load_from_disk(file_path)
        logging.info(f"Successfully loaded dataset with {len(dataset)} examples")
        df = pd.DataFrame(dataset)
        
        # Format columns as in curate_med_s1k_new.py
        df["Question"] = df["question"] + "\nWhat is the diagnosis of the patient?"
        df["Complex_CoT"] = df["thinking"]
        
        # Choose diagnosis based on priority order
        diagnosis_fields = [
            'diagnosis_final',
            'diagnosis_clinical_and_final',
            'diagnosis_pathological',
            'diagnosis_anatomical',
            'diagnosis_diagnosis_and_management',
            'diagnosis_diagnosis',
            'diagnosis_clinical',
            'diagnosis_laboratory',
            'diagnosis_psychiatric'
        ]
        
        def get_diagnosis(row):
            for field in diagnosis_fields:
                if field in row and pd.notna(row[field]):
                    return row[field]
            return "No diagnosis available"
            
        df["Response"] = df.apply(get_diagnosis, axis=1)
        
        # Keep only needed columns
        df = df[["Question", "Complex_CoT", "Response"]]
        return df
        
    except Exception as e:
        logging.error(f"Failed to load dataset from {file_path}: {e}")
        raise

async def analyze_sample_quality(sample: Dict) -> Dict:
    """Analyze the quality of a single sample using LLM"""
    prompt = f"""Analyze this medical question-answer pair for its quality as training data for an LLM. Return your analysis in the exact JSON format shown below, ensuring all fields are present and the JSON is complete and valid.

Question: {sample['Question']}

Complex Reasoning: {sample['Complex_CoT']}

Answer: {sample['Response']}

Required JSON format:
{{
    "has_table_refs": true/false,
    "has_figure_refs": true/false,
    "table_ref_quotes": ["quote 1", "quote 2"],
    "figure_ref_quotes": ["quote 1", "quote 2"],
    "has_confounding": true/false,
    "confounding_issues": ["issue 1", "issue 2"],
    "has_formatting_issues": true/false,
    "formatting_issues": ["issue 1", "issue 2"],
    "question_contains_answer": true/false,
    "is_coherent": true/false,
    "overall_quality": "high"/"medium"/"low",
    "explanation": "Detailed explanation here"
}}

Analyze the text for:
1. Table/figure references (e.g. "as shown in Table 1")
2. Confounding elements (unclear pronouns, missing context)
3. Punctuation/formatting issues
4. Whether question includes answer
5. Overall coherence and completeness

Ensure your response is ONLY the JSON object, with no additional text before or after."""

    # Get model response
    config = load_config()
    model_name = config["model_choices"]["curation"]
    response = await get_model_response(prompt, model=model_name)
    
    try:
        # Try to find JSON in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            analysis = json.loads(json_str)
            
            # Add original text for reference
            analysis["original_text"] = {
                "question": sample["Question"],
                "complex_cot": sample["Complex_CoT"],
                "response": sample["Response"]
            }
            return analysis
    except json.JSONDecodeError:
        logging.error(f"Failed to parse model response as JSON: {response}")
    
    # If we couldn't parse the JSON, return error with original text
    return {
        "error": "Failed to parse model response",
        "raw_response": response,
        "original_text": {
            "question": sample["Question"],
            "complex_cot": sample["Complex_CoT"],
            "response": sample["Response"]
        }
    }

async def analyze_dataset_quality(df: pd.DataFrame, n_samples: int = 100) -> List[Dict]:
    """Analyze quality of n_samples from the dataset"""
    # Sample randomly
    samples = df.sample(n=n_samples, random_state=42)
    
    # Create tasks
    tasks = []
    for _, sample in samples.iterrows():
        tasks.append(analyze_sample_quality(sample.to_dict()))
    
    # Run in parallel with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Analyzing samples") as pbar:
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            pbar.update(1)
    
    return results

async def main():
    # Setup logging
    setup_logging()
    
    # Load dataset
    logging.info("Loading NEJMCR dataset...")
    df = load_nejmcr_dataset()
    logging.info(f"Loaded {len(df)} examples")
    
    # Analyze samples
    logging.info("Starting quality analysis...")
    results = await analyze_dataset_quality(df, n_samples=100)
    
    # Save results
    output_path = os.path.join(MED_S1_DIR, "data", "nejmcr_quality_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved analysis results to {output_path}")
    
    # Print summary statistics
    n_table_refs = sum(1 for r in results if r.get("has_table_refs", False))
    n_figure_refs = sum(1 for r in results if r.get("has_figure_refs", False))
    n_confounding = sum(1 for r in results if r.get("has_confounding", False))
    n_formatting = sum(1 for r in results if r.get("has_formatting_issues", False))
    n_contains_answer = sum(1 for r in results if r.get("question_contains_answer", False))
    
    quality_counts = {
        "high": sum(1 for r in results if r.get("overall_quality") == "high"),
        "medium": sum(1 for r in results if r.get("overall_quality") == "medium"),
        "low": sum(1 for r in results if r.get("overall_quality") == "low")
    }
    
    print("\nAnalysis Summary:")
    print(f"Samples with table references: {n_table_refs}")
    print(f"Samples with figure references: {n_figure_refs}")
    print(f"Samples with confounding issues: {n_confounding}")
    print(f"Samples with formatting issues: {n_formatting}")
    print(f"Samples where question contains answer: {n_contains_answer}")
    print("\nQuality distribution:")
    for quality, count in quality_counts.items():
        print(f"{quality}: {count}")

if __name__ == "__main__":
    asyncio.run(main())