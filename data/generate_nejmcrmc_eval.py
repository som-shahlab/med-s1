import json
import argparse
import asyncio
from typing import Dict, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk
from utils.openai_utils import get_model_response
from curation_methods.clinical_formatting import (
    transform_to_nejmcr_qa,
    transform_to_nejmcr_reason,
    transform_to_nejmcr_clean
)

def load_config() -> Dict:
    """Load configuration from config.json"""
    import os
    assert os.getenv("MED_S1_DIR") is not None, "MED_S1_DIR environment variable not set"
    with open(os.path.join(os.getenv("MED_S1_DIR"), "config.json"), "r") as f:
        return json.load(f)

def load_nejmcr_dataset(config: Dict) -> List[Dict]:
    """Load NEJMCR dataset from config path."""
    dataset_config = config["train_datasets"]["nejmcr"]
    file_path = dataset_config["file_path"]
    
    print(f"\nLoading NEJMCR dataset from {file_path}")
    try:
        dataset = load_from_disk(file_path)
        print(f"Successfully loaded dataset with {len(dataset)} examples")
        
        # Convert to list of dicts
        cases = []
        for item in dataset:
            # Get diagnosis based on priority
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
            
            diagnosis = None
            for field in diagnosis_fields:
                if field in item and item[field]:
                    diagnosis = item[field]
                    break
            
            if not diagnosis:
                continue
                
            cases.append({
                'text': item['question'],
                'reasoning': item['thinking'],
                'diagnosis': diagnosis
            })
        
        print(f"Extracted {len(cases)} valid cases with diagnosis")
        return cases
    except Exception as e:
        print(f"Failed to load dataset from {file_path}: {e}")
        raise

def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Count tokens using the model's tokenizer."""
    return len(tokenizer(text).input_ids)

async def generate_mc_options(question: str, answer: str, reasoning: str, model_key: str = "gemini-2.0-flash", max_retries: int = 3) -> Optional[List[str]]:
    """Generate challenging multiple choice options."""
    prompt = f"""You are an expert medical educator. Generate three challenging but clearly incorrect options for this question.

Question:
{question}

Correct answer: {answer}

Reasoning that led to this answer:
{reasoning}

Generate three alternative options that:
1. Are medically plausible given the case presentation
2. E.g. could be reasonable differential diagnoses, treatment approaches, etiologies, etc.
3. Match the type of the correct answer (if it's a diagnosis, generate diagnoses; if treatment, generate treatments)
4. Could represent diagnostic or therapeutic pitfalls
5. Are categorically incorrect based on the full case details
6. Are similar in specificity and detail level to the correct answer
7. Are distinct from each other and the correct answer

IMPORTANT:
- Each option must be a complete, medically valid phrase
- Options must be clearly wrong but not obviously so
- Each option should test understanding of different aspects of the case
- Format as exactly three options, one per line, no numbering

Example for diagnosis case:
Correct: Acute bacterial endocarditis
Options:
Acute viral myocarditis
Aortic dissection with coronary involvement
Pericardial tamponade

Example for treatment case:
Correct: Immediate surgical debridement
Options:
Conservative management with oral antibiotics
Percutaneous needle aspiration
Local steroid injection
"""
    
    for attempt in range(max_retries):
        try:
            response = await get_model_response(prompt, model=model_key)
            if response:
                options = [line.strip() for line in response.split('\n') if line.strip()]
                
                # Validate options
                if len(options) == 3:
                    # Check for duplicates or too-similar options
                    option_set = set(opt.lower() for opt in options)
                    if len(option_set) == 3 and answer.lower() not in option_set:
                        # Check lengths are similar
                        ans_len = len(answer)
                        if all(abs(len(opt) - ans_len) < ans_len * 0.5 for opt in options):
                            # Verify each option is a proper medical phrase
                            if all(len(opt.split()) > 1 for opt in options):  # At least 2 words
                                return options
                
                if attempt < max_retries - 1:
                    prompt += "\n\nPrevious attempt failed validation. Please ensure:\n"
                    prompt += "1. Exactly three distinct medical options\n"
                    prompt += "2. Each option is a complete medical phrase\n"
                    prompt += "3. Options match the type of the correct answer\n"
                    prompt += "4. No duplicates or too-similar options"
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}: {str(e)}")
            else:
                print(f"Failed to generate valid options after {max_retries} attempts")
    
    return None

async def process_case(case: Dict, model_key: str) -> Optional[Dict]:
    """Process a single case into MC format."""
    # First generate question and answer
    qa_result = await transform_to_nejmcr_qa(
        question=case['text'],
        cot=case['reasoning'],
        answer=case['diagnosis'],
        model_key=model_key
    )
    
    if not qa_result:
        return None
        
    # Parse QA result
    qa_lines = qa_result.split('\n')
    question = None
    answer = None
    
    for line in qa_lines:
        if line.startswith('Question:'):
            question = line[9:].strip()
        elif line.startswith('Answer:'):
            answer = line[7:].strip()
    
    if not (question and answer):
        return None
    
    # Get clean reasoning
    reasoning = await transform_to_nejmcr_reason(
        cot=case['reasoning'],
        question=question,  # Use generated question
        answer=answer,      # Use generated answer
        model_key=model_key
    )
    
    if not reasoning:
        return None
        
    clean_reasoning = await transform_to_nejmcr_clean(reasoning, model_key)
    if not clean_reasoning:
        return None
    
    # Generate challenging options
    options = await generate_mc_options(
        question,  # Use generated question
        answer,    # Use generated answer
        clean_reasoning,
        model_key
    )
    
    if options:
        return {
            'question': question,  # Use generated question
            'options': {
                'A': answer,     # Use generated answer
                'B': options[0],
                'C': options[1],
                'D': options[2]
            },
            'answer_idx': 'A',
            'answer': answer     # Use generated answer
        }
    
    return None

async def process_batch(batch: List[Dict], model_key: str) -> List[Dict]:
    """Process a batch of cases concurrently."""
    # Create tasks for all cases in batch
    tasks = [process_case(case, model_key) for case in batch]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None values
    return [r for r in results if r]

async def generate_nejmcrmc(output_path: str, merge: bool = False, model_key: str = "gemini-2.0-flash", batch_size: int = 350):
    """Generate NEJMCRMC evaluation dataset."""
    # Load config and get tokenizer
    config = load_config()
    model_name = config["models"][config["model_choices"]["base"]]["hf_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load NEJMCR dataset from config
    nejmcr_data = load_nejmcr_dataset(config)
    
    # Filter for cases >8k tokens using same tokenization as training
    long_cases = []
    print("\nFiltering cases by token length...")
    for case in tqdm(nejmcr_data, desc="Counting tokens"):
        full_text = f"{case['text']} {case['reasoning']} {case['diagnosis']}"
        token_count = count_tokens(full_text, tokenizer)
        if token_count > 8192:  # Match threshold from curate_med_s1k_new.py
            long_cases.append(case)
    
    print(f"Found {len(long_cases)} cases with >8192 tokens")
    
    # Process cases in batches with concurrent API calls
    mc_questions = []
    
    for i in tqdm(range(0, len(long_cases), batch_size), desc="Processing cases"):
        batch = long_cases[i:i + batch_size]
        batch_results = await process_batch(batch, model_key)
        mc_questions.extend(batch_results)
    
    print(f"Generated {len(mc_questions)} multiple choice questions")
    
    # Print example for verification
    if mc_questions:
        print("\nExample MC question:")
        example = mc_questions[0]
        print(f"Question: {example['question']}")
        print("Options:")
        for k, v in example['options'].items():
            print(f"{k}: {v}")
        print(f"Answer: {example['answer']} (Index: {example['answer_idx']})")
    
    # Create NEJMCRMC dataset in expected format
    nejmcrmc_data = {
        "NEJMCRMC": mc_questions
    }
    
    # Save or merge
    if merge:
        with open(output_path, 'r') as f:
            eval_data = json.load(f)
        # Remove existing NEJMCRMC if present
        eval_data.pop('NEJMCRMC', None)
        # Add new NEJMCRMC data
        eval_data.update(nejmcrmc_data)
        with open(output_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f"Added/Updated NEJMCRMC in {output_path}")
    else:
        with open(output_path, 'w') as f:
            json.dump(nejmcrmc_data, f, indent=2)
        print(f"Saved NEJMCRMC data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate NEJMCRMC evaluation dataset')
    parser.add_argument('--eval_data', default='med-s1/eval/data/eval_data.json',
                      help='Path to eval_data.json to merge into')
    parser.add_argument('--output', default='med-s1/eval/data/nejmcrmc_eval.json',
                      help='Output path for separate NEJMCRMC dataset')
    parser.add_argument('--merge', action='store_true',
                      help='Add NEJMCRMC to eval_data.json')
    parser.add_argument('--model_key', default='gemini-2.0-flash',
                      help='Model to use for transformations')
    parser.add_argument('--batch_size', type=int, default=350,
                      help='Batch size for concurrent processing')
    
    args = parser.parse_args()
    output_path = args.eval_data if args.merge else args.output
    asyncio.run(generate_nejmcrmc(output_path, args.merge, args.model_key, args.batch_size))

if __name__ == '__main__':
    main()