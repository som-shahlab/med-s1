import json
import argparse
import asyncio
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils.openai_utils import get_model_response

# Sources to process
VALID_SOURCES = {
    'MedMCQA_validation',
    'MedQA_USLME_test',
    'PubMedQA_test',
    'MMLU-Pro_Medical_test',
    'GPQA_Medical_test'
}

async def is_care_planning_or_diagnosis(question: Dict, model_key: str = "gemini-2.0-flash") -> bool:
    """Use Gemini to determine if question relates to care planning or diagnosis."""
    prompt = f"""You are an expert medical educator. Determine if this medical question relates to care planning or diagnosis.

Question: {question['question']}
Options:
{chr(10).join(f"{k}: {v}" for k,v in question['options'].items())}
Answer: {question['answer']}

A question relates to care planning or diagnosis if it involves:
1. Making a diagnosis
2. Choosing between treatment options
3. Planning patient care
4. Clinical decision making
5. Therapeutic interventions
6. Disease management
7. Treatment selection
8. Prognosis assessment

The question must also ask about care planning or diagnosis for a specific patient.

Analyze the question carefully and respond with ONLY 'Yes' or 'No'.
"""
    
    try:
        response = await get_model_response(prompt, model=model_key)
        if response:
            return response.strip().lower() == 'yes'
    except Exception as e:
        print(f"Error classifying question: {e}")
    return False

async def process_batch(batch: List[Dict], model_key: str = "gemini-2.0-flash") -> List[Dict]:
    """Process a batch of questions concurrently."""
    # Create tasks for all questions in batch
    tasks = [is_care_planning_or_diagnosis(question, model_key) for question in batch]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Return filtered questions
    return [q for q, is_med in zip(batch, results) if is_med]

def create_nota_version(question: Dict) -> Dict:
    """Create NOTA (None of the Above) version of a question.
    
    Takes the correct answer option and moves it to the last position (D),
    replacing it with 'None of the above' and adjusting other options accordingly.
    """
    nota_question = question.copy()
    correct_idx = question['answer_idx']  # Get the correct option index (e.g., 'A', 'B', etc.)
    correct_answer = question['options'][correct_idx]  # Store the correct answer
    
    # Create new options dict
    new_options = {}
    old_options = question['options'].copy()
    del old_options[correct_idx]  # Remove correct answer
    
    # Fill options A-C with remaining options in order
    option_keys = sorted(old_options.keys())
    for new_idx, old_idx in zip(['A', 'B', 'C'], option_keys):
        new_options[new_idx] = old_options[old_idx]
    
    # Add 'None of the above' as option D
    new_options['D'] = 'None of the above'
    
    # Update the question
    nota_question['options'] = new_options
    nota_question['answer_idx'] = 'D'  # NOTA is always D
    nota_question['answer'] = 'None of the above'
    
    return nota_question

async def generate_medds(eval_data_path: str, output_path: str, merge: bool = False, batch_size: int = 350):
    """Generate MedDS evaluation dataset."""
    # Load existing eval data
    with open(eval_data_path, 'r') as f:
        eval_data = json.load(f)
    
    # Extract questions only from valid sources
    input_data = []
    for source in VALID_SOURCES:
        if source in eval_data:
            for q in eval_data[source]:
                q['source'] = source  # Ensure source is set
                input_data.append(q)
    
    print(f"\nProcessing {len(input_data)} questions from {len(VALID_SOURCES)} sources...")
    
    # Group by source for processing
    questions_by_source = {}
    for item in input_data:
        source = item['source']
        if source not in questions_by_source:
            questions_by_source[source] = []
        questions_by_source[source].append(item)
    
    # Process each source
    filtered_questions = []
    
    for source, questions in questions_by_source.items():
        print(f"\nProcessing {source}...")
        
        # Process in batches with concurrent API calls
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Classifying {source}"):
            batch = questions[i:i + batch_size]
            batch_filtered = await process_batch(batch)
            filtered_questions.extend(batch_filtered)
    
    print(f"\nFound {len(filtered_questions)} care planning/diagnosis questions")
    
    # Create MedDS and MedDS_NOTA datasets
    medds_data = {
        "MedDS": filtered_questions,
        "MedDS_NOTA": [create_nota_version(q) for q in filtered_questions]
    }
    
    # Print example for verification
    if filtered_questions:
        example = filtered_questions[0]
        nota_example = create_nota_version(example)
        print("\nExample conversion:")
        print("Original:")
        print(f"Source: {example['source']}")
        print(f"Question: {example['question']}")
        print("Options:")
        for k, v in example['options'].items():
            print(f"{k}: {v}")
        print(f"Answer: {example['answer']} (Index: {example['answer_idx']})")
        
        print("\nNOTA version:")
        print(f"Question: {nota_example['question']}")
        print("Options:")
        for k, v in nota_example['options'].items():
            print(f"{k}: {v}")
        print(f"Answer: {nota_example['answer']} (Index: {nota_example['answer_idx']})")
    
    # Save or merge
    if merge:
        # Remove existing MedDS sources if present
        eval_data.pop('MedDS', None)
        eval_data.pop('MedDS_NOTA', None)
        
        # Add new MedDS sources
        eval_data.update(medds_data)
        with open(eval_data_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f"Added/Updated MedDS and MedDS_NOTA in {eval_data_path}")
    else:
        # Save as separate file
        with open(output_path, 'w') as f:
            json.dump(medds_data, f, indent=2)
        print(f"Saved MedDS data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate MedDS evaluation dataset')
    parser.add_argument('--eval_data', default='med-s1/eval/data/eval_data.json',
                      help='Path to eval_data.json to merge into')
    parser.add_argument('--output', default='med-s1/eval/data/medds_eval.json',
                      help='Output path for separate MedDS dataset')
    parser.add_argument('--merge', action='store_true',
                      help='Add MedDS and MedDS_NOTA to eval_data.json')
    parser.add_argument('--model_key', default='gemini-2.0-flash',
                      help='Model to use for transformations')
    parser.add_argument('--batch_size', type=int, default=350,
                      help='Batch size for concurrent processing')
    
    args = parser.parse_args()
    output_path = args.eval_data if args.merge else args.output
    asyncio.run(generate_medds(output_path, args.merge, args.model_key, args.batch_size))

if __name__ == '__main__':
    main()