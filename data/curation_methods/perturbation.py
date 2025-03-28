"""
Perturbation methods for transforming step-based chain of thought reasoning.
"""

import random
import re
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd

async def parse_steps(cot: str) -> List[str]:
    """Parse steps from a step-based chain of thought reasoning."""
    # Split by step headers (## Step N:)
    steps = re.split(r'##\s*Step\s*\d+:', cot)
    # Remove empty strings and strip whitespace
    steps = [step.strip() for step in steps if step.strip()]
    return steps

def renumber_steps(steps: List[str]) -> str:
    """Renumber steps in the chain of thought."""
    result = ""
    for i, step in enumerate(steps, 1):
        result += f"## Step {i}:{step}\n\n"
    return result.strip()

async def collapse_consecutive_steps(cot: str, rate: float, model_key: str) -> str:
    """Collapse consecutive steps into single steps."""
    from utils.openai_utils import get_model_response
    
    steps = await parse_steps(cot)
    if len(steps) <= 1:
        return cot
        
    # Calculate how many times to apply collapse based on rate
    max_collapses = len(steps) - 1  # Maximum possible collapses
    n_collapses = int(rate * max_collapses)
    
    # Perform collapses
    for _ in range(n_collapses):
        if len(steps) <= 1:
            break
            
        # Randomly select consecutive pair
        idx = random.randint(0, len(steps)-2)
        
        # Create prompt to merge steps
        merge_prompt = f"""
Merge these two consecutive steps into a single concise step (<80 words) that preserves the key reasoning:

Step 1: {steps[idx]}

Step 2: {steps[idx+1]}

IMPORTANT: Return ONLY the merged step content, no introduction or explanation.
"""
        # Get merged step
        merged_step = await get_model_response(merge_prompt, model=model_key, max_tokens=200)
        if merged_step:
            # Replace consecutive steps with merged step
            steps = steps[:idx] + [merged_step] + steps[idx+2:]
    
    return renumber_steps(steps)

async def skip_steps(cot: str, rate: float) -> str:
    """Skip random steps in the chain of thought."""
    steps = await parse_steps(cot)
    if len(steps) <= 1:
        return cot
        
    # Calculate number of steps to skip
    n_skip = int(rate * len(steps))
    if n_skip >= len(steps):
        n_skip = len(steps) - 1  # Always keep at least one step
        
    # Randomly select steps to skip
    indices_to_keep = sorted(random.sample(range(len(steps)), len(steps) - n_skip))
    steps = [steps[i] for i in indices_to_keep]
    
    return renumber_steps(steps)

async def shuffle_steps(cot: str, rate: float) -> str:
    """Shuffle random steps in the chain of thought."""
    steps = await parse_steps(cot)
    if len(steps) <= 1:
        return cot
        
    # Calculate number of steps to shuffle
    n_shuffle = int(rate * len(steps))
    if n_shuffle > len(steps):
        n_shuffle = len(steps)
        
    # Select random indices to shuffle
    shuffle_indices = random.sample(range(len(steps)), n_shuffle)
    shuffle_steps = [steps[i] for i in shuffle_indices]
    # Shuffle selected steps
    random.shuffle(shuffle_steps)
    
    # Replace shuffled steps
    for new_step, idx in zip(shuffle_steps, shuffle_indices):
        steps[idx] = new_step
        
    return renumber_steps(steps)

async def add_irrelevant_steps(cot: str, rate: float, other_cots: List[str]) -> str:
    """Add irrelevant steps from other chain of thoughts."""
    if not other_cots:
        return cot
        
    steps = await parse_steps(cot)
    
    # Get random steps from other CoTs
    other_steps = []
    for other_cot in other_cots:
        other_steps.extend(await parse_steps(other_cot))
    
    if not other_steps:
        return cot
        
    # Calculate number of steps to add
    n_add = int(rate * len(steps))
    
    # Add random steps at random positions
    for _ in range(n_add):
        # Select random step from other CoTs
        new_step = random.choice(other_steps)
        # Insert at random position
        insert_pos = random.randint(0, len(steps))
        steps.insert(insert_pos, new_step)
    
    return renumber_steps(steps)

async def wrong_answer(cot: str, model_key: str, question: str, correct_response: str) -> Tuple[str, str]:
    """Change the last step and response to be categorically wrong."""
    steps = await parse_steps(cot)
    if not steps:
        return cot, correct_response
        
    from utils.openai_utils import get_model_response
    
    # Create prompt to generate wrong answer
    wrong_answer_prompt = f"""
Given this question and correct answer, generate a categorically wrong but plausible-sounding final step and response.

Question: {question}

Original final step:
{steps[-1]}

Original response:
{correct_response}

IMPORTANT: Return the new wrong final step AND new wrong response in this exact format:
STEP: [your wrong final step content]
RESPONSE: [your wrong response]
"""
    # Get wrong answer
    result = await get_model_response(wrong_answer_prompt, model=model_key, max_tokens=400)
    if result:
        # Parse step and response
        step_match = re.search(r'STEP:\s*(.*?)(?=RESPONSE:|$)', result, re.DOTALL)
        response_match = re.search(r'RESPONSE:\s*(.*?)$', result, re.DOTALL)
        
        if step_match:
            steps[-1] = step_match.group(1).strip()
        
        if response_match:
            new_response = response_match.group(1).strip()
        else:
            new_response = correct_response
            
        return renumber_steps(steps), new_response
    
    return cot, correct_response

async def apply_perturbation(
    df: pd.DataFrame,
    config: Dict,
    perturbation_type: str,
    rate: Optional[float] = None,
    model_key: Optional[str] = None
) -> pd.DataFrame:
    """Apply perturbation to the selected examples."""
    logging.info(f"Applying {perturbation_type} perturbation")
    
    # Validate rate if needed
    if perturbation_type != "answer" and rate not in [0.33, 0.66, 1.0]:
        raise ValueError(f"Invalid rate {rate} for {perturbation_type} perturbation")
    
    # Get selected examples
    selected_df = df[df['selected_for_training']].copy()
    
    # For add_irrelevant, we need other CoTs
    other_cots = None
    if perturbation_type == "add_irrelevant":
        other_cots = selected_df['Complex_CoT'].tolist()
    
    # Process each example
    for idx, row in selected_df.iterrows():
        cot = row['Complex_CoT']
        if pd.isna(cot) or not cot.strip():
            continue
            
        if perturbation_type == "collapse_consecutive":
            result = await collapse_consecutive_steps(cot, rate, model_key)
            if result:
                df.loc[idx, 'Complex_CoT'] = result
        elif perturbation_type == "skip":
            result = await skip_steps(cot, rate)
            if result:
                df.loc[idx, 'Complex_CoT'] = result
        elif perturbation_type == "shuffle":
            result = await shuffle_steps(cot, rate)
            if result:
                df.loc[idx, 'Complex_CoT'] = result
        elif perturbation_type == "add_irrelevant":
            # Remove current CoT from other_cots
            current_other_cots = [c for c in other_cots if c != cot]
            result = await add_irrelevant_steps(cot, rate, current_other_cots)
            if result:
                df.loc[idx, 'Complex_CoT'] = result
        elif perturbation_type == "answer":
            result, new_response = await wrong_answer(
                cot, model_key,
                row.get('Question', ''),
                row.get('Response', '')
            )
            if result:
                df.loc[idx, 'Complex_CoT'] = result
                df.loc[idx, 'Response'] = new_response
            
    return df