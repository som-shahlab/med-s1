"""
Qualitative evaluation of model outputs using LLM analysis.
"""

import os
import json
import random
import asyncio
import logging
import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from functools import lru_cache
from analysis_helpers import get_eval_example, score_example
from scorer import match_choice
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils.openai_utils import get_model_response

# Get MED_S1_DIR from environment
MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

FAILURE_MODE_PROMPT = """
Analyze the following model output and identify which failure modes are present. For each failure mode that applies, explain why with specific quotes from the output.

Question: {question}

Answer Options:
{options}

Correct Answer: {answer}

Model Output:
{output}

Failure Modes to Consider:
1. bad-formatting-and-wrong: Output doesn't have a clear letter choice (A-N) in the final answer AND the content is incorrect
2. bad-formatting-but-correct: Output doesn't have a clear letter choice but the content/reasoning is actually correct
3. premature-response: Gives answer before reasoning
4. medical-inaccuracy: Includes a statement that is categorically false
5. reasoning-fallacy: Commits a logical fallacy in the reasoning that is not just a factual inaccuracy
6. irrelevant-or-extraneous-content: Includes additional information or digressions not pertinent to the question
7. low-confidence: None of the answers are selected with confidence
8. missing-headers: Doesn't have the expected chat format headers (think and answer)
9. narrowed-but-excluded: Correctly narrowed down to include the right answer but then excluded it
10. narrow-consideration: Does not consider an important case or possibility
11. hallucinate-context: Hallucinated details about the input that weren't provided
12. bad-question: The input question lacks critical information needed to arrive at a definitive answer (e.g., missing key symptoms, test results, or context that would be necessary for diagnosis)
13. superlative: Uses a superlative like "only" or "all" that is categorically untrue
14. no-support: Makes a critical statement without offering support or rationale
15. commonality-bias: Choose what is most common when commonality is not relevant to the answer

For each applicable failure mode, provide:
1. A clear explanation of why it applies
2. Relevant quotes from the output
3. Impact on the final answer

IMPORTANT: Format your response as a JSON object with failure modes as keys and objects containing explanation, quotes, and impact as values.
"""

REASONING_QUALITY_PROMPT = """
Analyze the following chain-of-thought reasoning for a medical question:

CONTEXT:
Question: {question_text}

Answer Options:
{options}

Ground Truth Answer: {ground_truth}

Model's Chain-of-thought:
{chain_of_thought}

Please provide a detailed analysis in the following JSON format:

{{
  "DATA INTERPRETATION": {{
    "Score": "(0-10)",
    "Errors": "Detailed explanation of any errors in interpreting the given information..."
  }},
  "DIAGNOSTIC REASONING": {{
    "Score": "(0-10)",
    "Errors": "Analysis of the reasoning process and any logical flaws..."
  }},
  "INFORMATION GATHERING": {{
    "Score": "(0-10)",
    "Errors": "Assessment of how well relevant information was collected and used..."
  }},
  "CLINICAL JUDGMENT": {{
    "Score": "(0-10)",
    "Errors": "Evaluation of clinical decision-making and prioritization..."
  }},
  "PRACTICAL APPLICATION": {{
    "Score": "(0-10)",
    "Errors": "Analysis of how well clinical knowledge was applied to the case..."
  }},
  "overall": {{
    "Score": "(0-10)",
    "summary": "Overall assessment highlighting the most critical errors and strengths..."
  }}
}}

For each category:
- Score should be 0-10 (0 = completely flawed, 10 = completely correct)
- Errors should explain any issues found, even for high scores
- Be specific and cite examples from the chain-of-thought
"""

# Cache config loading
@lru_cache(maxsize=1)
def load_full_config():
    with open(os.path.join(MED_S1_DIR, "config.json"), 'r') as config_file:
        return json.load(config_file)

def format_options(options: Dict[str, str]) -> str:
    """Format options dictionary into readable string."""
    return "\n".join(f"{k}. {v}" for k, v in options.items())

async def analyze_failure_modes_batch(examples: List[Dict], model_key: str) -> List[Dict]:
    """Analyze failure modes for a batch of examples using LLM."""
    # Create tasks for all examples
    tasks = []
    for example in examples:
        prompt = FAILURE_MODE_PROMPT.format(
            question=example['question'],
            options=format_options(example['options']),
            answer=example['answer'],
            output=example['output']
        )
        tasks.append(get_model_response(prompt, model=model_key, max_tokens=8192))
    
    # Process batch with concurrent API calls
    batch_results = await asyncio.gather(*tasks)
    
    # Parse results
    parsed_results = []
    for example, result in zip(examples, batch_results):
        try:
            if result is None:
                logging.error("LLM API call failed")
                parsed_results.append({
                    "example": example,
                    "analysis": {}
                })
                continue
                
            # Try to parse as JSON first
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")
            
            parsed_results.append({
                "example": example,
                "analysis": parsed
            })
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse LLM output: {str(e)}\nOutput: {result}")
            parsed_results.append({
                "example": example,
                "analysis": {}
            })
    
    return parsed_results

async def analyze_reasoning_quality_batch(examples: List[Dict], model_key: str) -> List[Dict]:
    """Analyze reasoning quality for a batch of examples using LLM."""
    # Create tasks for all examples
    tasks = []
    for example in examples:
        prompt = REASONING_QUALITY_PROMPT.format(
            question_text=example['question'],
            options=format_options(example['options']),
            ground_truth=example['answer'],
            chain_of_thought=example['output']
        )
        tasks.append(get_model_response(prompt, model=model_key, max_tokens=8192))
    
    # Process batch with concurrent API calls
    batch_results = await asyncio.gather(*tasks)
    
    # Parse results
    parsed_results = []
    for example, result in zip(examples, batch_results):
        try:
            if result is None:
                logging.error("LLM API call failed")
                parsed_results.append({
                    "example": example,
                    "analysis": {}
                })
                continue
                
            # Try to parse as JSON first
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")
            
            # Normalize field names
            normalized = {}
            field_mapping = {
                "data_interpretation": ["data_interpretation", "DATA INTERPRETATION", "Data Interpretation"],
                "diagnostic_reasoning": ["diagnostic_reasoning", "DIAGNOSTIC REASONING", "Diagnostic Reasoning"],
                "information_gathering": ["information_gathering", "INFORMATION GATHERING", "Information Gathering"],
                "clinical_judgment": ["clinical_judgment", "CLINICAL JUDGMENT", "Clinical Judgment"],
                "practical_application": ["practical_application", "PRACTICAL APPLICATION", "Practical Application"],
                "overall": ["overall", "OVERALL", "Overall"]
            }
            
            # Find matching fields and normalize them
            for norm_name, variants in field_mapping.items():
                found = None
                for variant in variants:
                    if variant in parsed:
                        found = parsed[variant]
                        break
                if found is None:
                    raise ValueError(f"Missing required field: {norm_name}")
                normalized[norm_name] = found
            
            # Normalize score and errors fields
            for field, content in normalized.items():
                if field == "overall":
                    score_key = next((k for k in ["score", "Score"] if k in content), None)
                    summary_key = next((k for k in ["summary", "Summary"] if k in content), None)
                    if not score_key or not summary_key:
                        raise ValueError(f"Missing score or summary in overall field")
                    normalized[field] = {
                        "score": float(content[score_key]),
                        "summary": content[summary_key]
                    }
                else:
                    score_key = next((k for k in ["score", "Score"] if k in content), None)
                    errors_key = next((k for k in ["errors", "Errors"] if k in content), None)
                    if not score_key or not errors_key:
                        raise ValueError(f"Missing score or errors in {field}")
                    normalized[field] = {
                        "score": float(content[score_key]),
                        "errors": content[errors_key]
                    }
            
            parsed = normalized
            
            parsed_results.append({
                "example": example,
                "analysis": parsed
            })
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse LLM output: {str(e)}\nOutput: {result}")
            parsed_results.append({
                "example": example,
                "analysis": {}
            })
    
    return parsed_results

def load_results_json() -> dict:
    """Load the results.json file."""
    with open("/share/pi/nigam/users/calebwin/med-s1/results.json", 'r') as f:
        return json.load(f)

async def get_random_examples(experiment_name: str, n: int, correct: bool, seed: int = 42) -> List[Dict]:
    """Get n random examples that are either correct or incorrect."""
    from tqdm import tqdm
    random.seed(seed)
    
    # Get eval outputs file path
    results = load_results_json()
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    
    exp_results = results["experiments"][experiment_name]
    if "results" not in exp_results or "eval" not in exp_results["results"]:
        raise ValueError(f"No eval results found for {experiment_name}")
    
    outputs_path = exp_results["results"]["eval"]["outputs_path"]
    if not os.path.exists(outputs_path):
        raise ValueError(f"Eval outputs file not found: {outputs_path}")
    
    # Load all examples at once
    with open(outputs_path, 'r') as f:
        all_data = json.load(f)
    
    # Score all examples in parallel
    examples = []
    indices = []
    with tqdm(total=len(all_data), desc="Scoring examples") as pbar:
        for i, example in enumerate(all_data):
            is_correct = score_example(example)
            if is_correct == correct:
                examples.append(example)
                indices.append(i)
            pbar.update(1)
    
    # Randomly sample n examples
    if len(examples) > n:
        selected_indices = random.sample(range(len(examples)), n)
        examples = [examples[i] for i in selected_indices]
        indices = [indices[i] for i in selected_indices]
    elif len(examples) < n:
        logging.warning(f"Only found {len(examples)} {correct} examples, wanted {n}")
    
    return examples[:n], indices[:n]

async def run_qualitative_analysis(experiment_name: str, seed: int = 42) -> Dict[str, Any]:
    """Run full qualitative analysis on an experiment."""
    logging.info(f"Starting qualitative analysis for experiment {experiment_name}")
    
    # Get examples in parallel
    logging.info("Getting random examples...")
    incorrect_examples, incorrect_indices = await get_random_examples(experiment_name, 100, False, seed)
    correct_examples, correct_indices = await get_random_examples(experiment_name, 100, True, seed)
    
    # Get model key from config
    config = load_full_config()
    model_key = config.get("model_choices", {}).get("eval_qualitative", "gemini-2.0-flash")
    
    # Process in batches
    batch_size = 25  # Increased for better throughput
    total_incorrect_batches = (len(incorrect_examples) + batch_size - 1) // batch_size
    total_correct_batches = (len(correct_examples) + batch_size - 1) // batch_size
    
    logging.info(f"Processing {len(incorrect_examples)} incorrect examples in {total_incorrect_batches} batches")
    logging.info(f"Processing {len(correct_examples)} correct examples in {total_correct_batches} batches")
    
    results = {
        "metadata": {
            "experiment": experiment_name,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model": model_key,
            "seed": seed,
            "incorrect_indices": incorrect_indices,
            "correct_indices": correct_indices
        },
        "failure_mode_analysis": [],
        "reasoning_quality_analysis": {
            "correct": []
        }
    }
    
    # Analyze failure modes for incorrect examples
    from tqdm import tqdm
    logging.info("Analyzing failure modes...")
    with tqdm(total=len(incorrect_examples), desc="Analyzing failure modes") as pbar:
        for i in range(0, len(incorrect_examples), batch_size):
            batch = incorrect_examples[i:i+batch_size]
            batch_results = await analyze_failure_modes_batch(batch, model_key)
            results["failure_mode_analysis"].extend(batch_results)
            pbar.update(len(batch))

    # Analyze reasoning quality for correct examples only
    logging.info("Analyzing reasoning quality...")
    with tqdm(total=len(correct_examples), desc="Analyzing reasoning quality") as pbar:
        for i in range(0, len(correct_examples), batch_size):
            batch = correct_examples[i:i+batch_size]
            batch_results = await analyze_reasoning_quality_batch(batch, model_key)
            results["reasoning_quality_analysis"]["correct"].extend(batch_results)
            pbar.update(len(batch))
    
    # Calculate aggregate scores
    def aggregate_scores(analyses: List[Dict]) -> Dict[str, float]:
        scores = {
            "data_interpretation": [],
            "diagnostic_reasoning": [],
            "information_gathering": [],
            "clinical_judgment": [],
            "practical_application": [],
            "overall": []
        }
        
        for result in analyses:
            analysis = result["analysis"]
            for category in scores.keys():
                if category == "overall":
                    if category in analysis and "score" in analysis[category]:
                        scores[category].append(analysis[category]["score"])
                else:
                    if category in analysis and "score" in analysis[category]:
                        scores[category].append(analysis[category]["score"])
        
        return {k: sum(v)/len(v) if v else 0 for k, v in scores.items()}
    
    # Only aggregate scores for correct examples
    results["aggregate_scores"] = {
        "correct": aggregate_scores(results["reasoning_quality_analysis"]["correct"])
    }
    
    # Count failure modes
    failure_mode_counts = {}
    for result in results["failure_mode_analysis"]:
        for mode in result["analysis"].keys():
            failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1
    
    results["failure_mode_counts"] = failure_mode_counts
    
    logging.info("Qualitative analysis complete")
    return results

def add_qualitative_results(experiment_name: str, results: Dict[str, Any]):
    """Add qualitative analysis results to results.json."""
    results_path = os.path.join(MED_S1_DIR, "results.json")
    
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    # Save results file
    results_dir = os.path.join(MED_S1_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{experiment_name}_qualitative.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Update results.json
    if experiment_name in all_results["experiments"]:
        if "results" not in all_results["experiments"][experiment_name]:
            all_results["experiments"][experiment_name]["results"] = {}
        
        all_results["experiments"][experiment_name]["results"]["eval_qualitative"] = {
            "path": results_file,
            "timestamp": results["metadata"]["timestamp"]
        }
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

def get_extreme_reasoning_examples(
    experiment_name: str,
    category: str,
    n: int = 5,
    high: bool = True,
    print_format: bool = False
) -> Optional[List[Dict]]:
    """Get examples with extreme (high/low) scores in a reasoning category.
    
    Args:
        experiment_name: Name of experiment to analyze
        category: Reasoning category to analyze (data_interpretation, diagnostic_reasoning, etc.)
        n: Number of examples to return
        high: If True, get highest scoring examples; if False, get lowest scoring
        print_format: If True, prints examples instead of returning them
        
    Returns:
        List of dictionaries containing:
        - example: Original example from correct examples only
        - analysis: Reasoning quality analysis for this example
        - score: Score for the specified category
        
    Example:
        >>> # Get top 5 examples of good diagnostic reasoning
        >>> get_extreme_reasoning_examples(
        ...     "medqa-1k-random",
        ...     "diagnostic_reasoning",
        ...     n=5,
        ...     high=True,
        ...     print_format=True
        ... )
        Top 5 Diagnostic Reasoning Examples
        ===============================
        Example 1 (Score: 9.8):
        Question: ...
        ...
    """
    # Load results
    results_path = os.path.join(MED_S1_DIR, "results.json")
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    if experiment_name not in all_results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    
    exp_results = all_results["experiments"][experiment_name]
    if "results" not in exp_results or "eval_qualitative" not in exp_results["results"]:
        raise ValueError(f"No qualitative analysis results found for {experiment_name}")
    
    qual_path = exp_results["results"]["eval_qualitative"]["path"]
    if not os.path.exists(qual_path):
        raise ValueError(f"Qualitative results file not found: {qual_path}")
    
    with open(qual_path, 'r') as f:
        qual_results = json.load(f)
    
    # Get examples with their scores (only correct examples have reasoning analysis)
    examples = []
    for result in qual_results["reasoning_quality_analysis"]["correct"]:
        if category in result["analysis"] and "score" in result["analysis"][category]:
            examples.append({
                "example": result["example"],
                "analysis": result["analysis"],
                "score": result["analysis"][category]["score"]
            })
    
    if not examples:
        if print_format:
            print(f"No examples found for category: {category}")
            return None
        return []
    
    # Sort by score
    examples.sort(key=lambda x: x["score"], reverse=high)
    examples = examples[:n]
    
    if print_format:
        direction = "Top" if high else "Bottom"
        print(f"{direction} {n} {category.replace('_', ' ').title()} Examples")
        print("=" * (len(category) + n + 15))
        
        for i, item in enumerate(examples, 1):
            print(f"\nExample {i} (Score: {item['score']:.1f}):")
            print("Question:", item["example"]["question"])
            print("\nOptions:")
            for k, v in item["example"]["options"].items():
                print(f"{k}. {v}")
            print("\nCorrect Answer:", item["example"]["answer"])
            print("\nModel Output:", item["example"]["output"])
            print("\nAnalysis:")
            print(f"{category.replace('_', ' ').title()} Errors:")
            for error in item["analysis"][category]["errors"]:
                print(f"- {error}")
            print("\nOverall Summary:", item["analysis"]["overall"]["summary"])
            print("\n" + "-" * 80)
        
        return None
    
    return examples

def get_failure_mode_examples(experiment_name: str, failure_mode: str, print_format: bool = False) -> Optional[List[Dict]]:
    """Get examples of a specific failure mode with their analysis.
    
    Args:
        experiment_name: Name of experiment to analyze
        failure_mode: Name of failure mode to get examples for
        print_format: If True, prints examples instead of returning them
        
    Returns:
        List of dictionaries containing:
        - example: Original example that exhibited the failure mode
        - analysis: Analysis of the failure mode for this example
        
    Example:
        >>> get_failure_mode_examples("medqa-1k-random", "medical-inaccuracy", print_format=True)
        Medical Inaccuracy Examples
        =========================
        Example 1:
        Question: ...
        Options:
        A. ...
        B. ...
        ...
        
        Model Output:
        ...
        
        Analysis:
        Explanation: ...
        Quotes: ...
        Impact: ...
        
        Example 2:
        ...
    """
    # Load results
    results_path = os.path.join(MED_S1_DIR, "results.json")
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    if experiment_name not in all_results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    
    exp_results = all_results["experiments"][experiment_name]
    if "results" not in exp_results or "eval_qualitative" not in exp_results["results"]:
        raise ValueError(f"No qualitative analysis results found for {experiment_name}")
    
    qual_path = exp_results["results"]["eval_qualitative"]["path"]
    if not os.path.exists(qual_path):
        raise ValueError(f"Qualitative results file not found: {qual_path}")
    
    with open(qual_path, 'r') as f:
        qual_results = json.load(f)
    
    # Get examples with this failure mode
    examples = []
    for result in qual_results["failure_mode_analysis"]:
        if failure_mode in result["analysis"]:
            examples.append({
                "example": result["example"],
                "analysis": result["analysis"][failure_mode]
            })
    
    if not examples:
        if print_format:
            print(f"No examples found for failure mode: {failure_mode}")
            return None
        return []
    
    if print_format:
        print(f"{failure_mode.replace('-', ' ').title()} Examples")
        print("=" * (len(failure_mode) + 9))
        
        for i, item in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print("Question:", item["example"]["question"])
            print("\nOptions:")
            for k, v in item["example"]["options"].items():
                print(f"{k}. {v}")
            print("\nCorrect Answer:", item["example"]["answer"])
            print("\nModel Output:", item["example"]["output"])
            print("\nAnalysis:")
            print("Explanation:", item["analysis"]["explanation"])
            print("Quotes:", "\n".join(f"- {quote}" for quote in item["analysis"]["quotes"]))
            print("Impact:", item["analysis"]["impact"])
            print("\n" + "-" * 80)
        
        return None
    
    return examples

def analyze_qualitative_results(experiment_name: str, print_format: bool = False) -> Optional[Dict[str, Any]]:
    """Load and analyze qualitative evaluation results.
    
    Args:
        experiment_name: Name of experiment to analyze
        print_format: If True, prints formatted analysis instead of returning dict
        
    Returns:
        Dictionary containing:
        - metadata: Original metadata from analysis
        - failure_modes: Summary of failure mode frequencies from incorrect examples
        - reasoning_scores: Summary of reasoning quality scores from correct examples
        - example_counts: Number of examples analyzed in each category
        
    Example:
        >>> analyze_qualitative_results("medqa-1k-random", print_format=True)
        Qualitative Analysis Results for medqa-1k-random
        ===============================================
        Analysis performed: 2025-04-06T12:03:58.123456
        Model used: gemini-2.0-flash
        
        Failure Mode Analysis (100 incorrect examples)
        -------------------------------------------
        bad-formatting-and-wrong: 23 (23.0%)
        medical-inaccuracy: 45 (45.0%)
        ...
        
        Reasoning Quality Scores (Correct Examples)
        -----------------------------------
        Based on 100 correct examples:
          Data Interpretation: 8.1
          Diagnostic Reasoning: 7.9
          ...
    """
    # Load results
    results_path = os.path.join(MED_S1_DIR, "results.json")
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    if experiment_name not in all_results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    
    exp_results = all_results["experiments"][experiment_name]
    if "results" not in exp_results or "eval_qualitative" not in exp_results["results"]:
        raise ValueError(f"No qualitative analysis results found for {experiment_name}")
    
    qual_path = exp_results["results"]["eval_qualitative"]["path"]
    if not os.path.exists(qual_path):
        raise ValueError(f"Qualitative results file not found: {qual_path}")
    
    with open(qual_path, 'r') as f:
        qual_results = json.load(f)
    
    # Analyze results
    analysis = {
        "metadata": qual_results["metadata"],
        "failure_modes": {
            "counts": qual_results["failure_mode_counts"],
            "percentages": {
                mode: count * 100 / len(qual_results["failure_mode_analysis"])
                for mode, count in qual_results["failure_mode_counts"].items()
            },
            "examples": {
                mode: [
                    result["example"]
                    for result in qual_results["failure_mode_analysis"]
                    if mode in result["analysis"]
                ]
                for mode in qual_results["failure_mode_counts"].keys()
            }
        },
        "reasoning_scores": qual_results["aggregate_scores"]["correct"],
        "example_counts": {
            "incorrect": len(qual_results["failure_mode_analysis"]),
            "correct": len(qual_results["reasoning_quality_analysis"]["correct"])
        }
    }
    
    if print_format:
        print(f"Qualitative Analysis Results for {experiment_name}")
        print("=" * (37 + len(experiment_name)))
        print(f"Analysis performed: {analysis['metadata']['timestamp']}")
        print(f"Model used: {analysis['metadata']['model']}")
        print()
        
        print(f"Failure Mode Analysis ({analysis['example_counts']['incorrect']} incorrect examples)")
        print("-" * 50)
        for mode, count in sorted(analysis["failure_modes"]["counts"].items(),
                                key=lambda x: x[1], reverse=True):
            percentage = analysis["failure_modes"]["percentages"][mode]
            print(f"{mode}: {count} ({percentage:.1f}%)")
        print()
        
        print("Reasoning Quality Scores (Correct Examples)")
        print("-" * 35)
        
        categories = [
            "data_interpretation", "diagnostic_reasoning",
            "information_gathering", "clinical_judgment",
            "practical_application", "overall"
        ]
        
        n = analysis["example_counts"]["correct"]
        print(f"Based on {n} correct examples:")
        for category in categories:
            score = analysis["reasoning_scores"]["correct"][category]
            print(f"  {category.replace('_', ' ').title()}: {score:.1f}")
        print()
        
        return None
    
    return analysis

async def main():
    """Run qualitative analysis from command line."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Name of experiment to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.analyze:
        analyze_qualitative_results(args.experiment, print_format=True)
    else:
        results = await run_qualitative_analysis(args.experiment, args.seed)
        add_qualitative_results(args.experiment, results)

if __name__ == "__main__":
    asyncio.run(main())