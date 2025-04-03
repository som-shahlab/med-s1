"""
Perturbation methods for transforming step-based chain of thought reasoning.
"""

import random
import re
import logging
from typing import List, Dict, Optional, Tuple
import pandas as pd
import asyncio

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
        result += f"## Step {i}: {step}\n\n"
    return result.strip()

async def restore_reasoning(cot: str, model_key: str) -> Optional[str]:
    """Restore a perturbed reasoning trace to high-quality step-by-step format."""
    from utils.openai_utils import get_model_response
    
    restore_prompt = f"""
You are an expert medical educator. Your task is to improve this low-quality chain of thought reasoning and extract a high-quality step-by-step chain of thought reasoning.

The chain of thought may have issues like:
- Missing details
- Out of order steps
- Irrelevant details
- Brevity

Your task is to transform the following chain of thought reasoning into a clear, step-by-step format, adding steps or detail where appropriate.

Each step should:
1. Be numbered and have a clear title (e.g., "## Step 1: Assess the patient's condition")
2. Be organized in a logical sequence
5. Add detail where appropriate

Here's the chain of thought reasoning to transform:

{cot}

IMPORTANT: Return ONLY the improved step-by-step reasoning trace, starting directly with "## Step 1:". Do not include any introduction or explanation.
"""
    try:
        result = await get_model_response(restore_prompt, model=model_key, max_tokens=8192)
        
        # Validate result
        if not result or len(result.strip()) < 10:
            logging.warning("Restoration produced empty or very short result")
            return None
            
        # Ensure proper formatting
        step1_match = re.search(r'## Step 1:', result)
        if step1_match:
            result = result[step1_match.start():]
            
        # Validate step structure
        steps = re.findall(r'##\s*Step\s*\d+:', result)
        if len(steps) < 2:
            logging.warning("Restoration result has fewer than 2 steps")
            return None
            
        # Ensure result is longer than input (should add details)
        if len(result) <= len(cot):
            logging.warning("Restoration did not expand the content as expected")
            return None
            
        return result
    except Exception as e:
        logging.error(f"Error in restore_reasoning: {str(e)}")
        return None

async def collapse_consecutive_steps(cot: str, rate: float, model_key: str) -> str:
    """Collapse consecutive steps into single steps."""
    from utils.openai_utils import get_model_response
    
    steps = await parse_steps(cot)
    if len(steps) <= 1:
        return cot
        
    # Calculate how many times to apply collapse based on rate
    max_collapses = len(steps) - 1  # Maximum possible collapses
    if rate >= 1.0:
        # For rates > 1, collapse all steps (n_collapses = max_collapses)
        n_collapses = max_collapses
    else:
        n_collapses = int(min(rate + 0.01, 1.0) * max_collapses)
    
    # Prepare all merges at once
    if len(steps) <= 1:
        return cot
        
    # Select all pairs to merge upfront
    available_indices = list(range(len(steps)-1))
    merge_indices = []
    steps_to_merge = min(n_collapses, len(available_indices))
    
    # Select indices for merging
    if rate >= 1.0:
        # For rate >= 1.0, merge all consecutive pairs to collapse to single step
        merge_indices = list(range(0, len(steps)-1))
    else:
        # For other rates, avoid overlaps
        while len(merge_indices) < steps_to_merge and available_indices:
            idx = random.choice(available_indices)
            merge_indices.append(idx)
            # Remove selected index and adjacent indices to avoid overlaps
            available_indices = [i for i in available_indices
                               if abs(i - idx) > 1]
        
    def count_words(text: str) -> int:
        """Count the number of words in a string."""
        return len(text.split())
    
    # Create all merge prompts
    merge_tasks = []
    for idx in merge_indices:
        # Calculate average word count of the two steps
        step1_words = count_words(steps[idx])
        step2_words = count_words(steps[idx+1])
        avg_words = int((step1_words + step2_words) / 2)
        
        merge_prompt = f"""
Merge these two consecutive steps into a single concise step (<{avg_words} words) that preserves the key reasoning:

## Step 1: {steps[idx]}

## Step 2: {steps[idx+1]}

IMPORTANT: Return ONLY the merged step including step title and content, no introduction or explanation.

## Step 1+2:
"""
        merge_tasks.append(get_model_response(merge_prompt, model=model_key, max_tokens=200))
    
    # Execute all merges concurrently
    merged_results = await asyncio.gather(*merge_tasks)
    
    # Apply merges in reverse order to maintain correct indices
    for idx, merged_step in sorted(zip(merge_indices, merged_results), reverse=True):
        if merged_step:
            steps = steps[:idx] + [merged_step.replace("## Step 1+2:", "").strip()] + steps[idx+2:]
    
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
The wrong answer should be clearly incorrect to experts but not obviously different in style.

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
    model_key: Optional[str] = None,
    restore: bool = False
) -> pd.DataFrame:
    """Apply perturbation to the selected examples."""
    logging.info(f"Applying {perturbation_type} perturbation")
    
    # Validate rate
    if rate not in [0.33, 0.66, 1.0]:
        raise ValueError(f"Invalid rate {rate} for {perturbation_type} perturbation")
    
    # Get selected examples
    selected_df = df[df['selected_for_training']].copy()
    
    # For answer perturbation, select exact number of examples to perturb
    perturb_indices = None
    if perturbation_type == "answer":
        n_selected = len(selected_df)
        n_perturb = int(rate * n_selected)
        perturb_indices = set(random.sample(selected_df.index.tolist(), n_perturb))
        logging.info(f"Selected {n_perturb} examples for wrong answer perturbation")
    
    # For add_irrelevant, we need other CoTs
    other_cots = None
    if perturbation_type == "add_irrelevant":
        other_cots = selected_df['Complex_CoT'].tolist()
    
    # Process examples in parallel batches
    # Since we're I/O bound (waiting for API calls), we can be aggressive with parallelism
    batch_size = 500  # Match step_extraction.py's batch size
    tasks = []
    
    async def process_example(idx: int, row: pd.Series) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
        # First get the true original content before any modifications
        orig = row.get('Complex_CoT_orig', row['Complex_CoT'])  # Get original if available
        
        # Then get input for perturbation from extracted version
        if 'Complex_CoT_extracted' in row:
            cot = row['Complex_CoT_extracted']
            logging.info(f"Using extracted version for perturbation input")
        else:
            cot = orig
            logging.info(f"No extracted version found, using original for perturbation input")
        
        if pd.isna(cot) or not cot.strip():
            logging.warning(f"Example {idx} has empty input for perturbation")
            return idx, orig, None, None
            
        # Apply perturbation
        if perturbation_type == "collapse_consecutive":
            result = await collapse_consecutive_steps(cot, rate, model_key)
        elif perturbation_type == "skip":
            result = await skip_steps(cot, rate)
        elif perturbation_type == "shuffle":
            result = await shuffle_steps(cot, rate)
        elif perturbation_type == "add_irrelevant":
            current_other_cots = [c for c in other_cots if c != cot]
            result = await add_irrelevant_steps(cot, rate, current_other_cots)
        elif perturbation_type == "answer":
            # Only apply wrong answer to selected indices
            if perturb_indices and idx in perturb_indices:
                result, new_response = await wrong_answer(
                    cot, model_key,
                    row.get('Question', ''),
                    row.get('Response', '')
                )
                return idx, cot, result, new_response
            else:
                return idx, cot, cot, row.get('Response', '')
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
            
        if not result:
            return idx, None, None, None
            
        # If restore is True, restore the perturbed CoT
        restored = None
        if restore and result:
            restored = await restore_reasoning(result, model_key)
            
        return idx, cot, result, restored
    
    # Create tasks for all examples
    for idx, row in selected_df.iterrows():
        tasks.append(process_example(idx, row))
        
        # When we have enough tasks or this is the last batch
        if len(tasks) >= batch_size or idx == selected_df.index[-1]:
            # Process batch
            results = await asyncio.gather(*tasks)
            
            # Log column state before updates
            logging.info(f"Before updates - Columns: {df.columns.tolist()}")
            
            # Update DataFrame with results
            for idx, orig, perturbed, restored in results:
                # Track what version we're using
                input_version = "extracted" if 'Complex_CoT_extracted' in df.columns else "complex_cot"
                logging.info(f"Example {idx} using {input_version} version as input")
                
                # Handle original content
                if orig is not None and 'Complex_CoT_orig' not in df.columns:
                    df.loc[idx, 'Complex_CoT_orig'] = orig
                    logging.info(f"Preserved original content for example {idx}")
                elif orig is not None:
                    logging.info(f"Complex_CoT_orig already exists, keeping existing content")
                
                # Handle perturbed content
                if perturbed is not None:
                    df.loc[idx, 'Complex_CoT_perturbed'] = perturbed
                    logging.info(f"Stored perturbed version for example {idx}")
                    
                    # Update Complex_CoT based on experiment type
                    if restore:
                        if restored is not None:
                            df.loc[idx, 'Complex_CoT_restored'] = restored
                            df.loc[idx, 'Complex_CoT'] = restored
                            logging.info(f"Using restored version as final for example {idx}")
                        else:
                            # Fallback to extracted version if available
                            if 'Complex_CoT_extracted' in df.columns:
                                df.loc[idx, 'Complex_CoT'] = df.loc[idx, 'Complex_CoT_extracted']
                                logging.info(f"Restoration failed, using extracted version for example {idx}")
                            else:
                                logging.warning(f"No extracted version available for example {idx}, using original")
                                df.loc[idx, 'Complex_CoT'] = orig
                    else:
                        # For perturbation-only, use perturbed version
                        df.loc[idx, 'Complex_CoT'] = perturbed
                        logging.info(f"Using perturbed version as final for example {idx}")
                else:
                    logging.warning(f"Perturbation failed for example {idx}")
            
            # Log final state
            logging.info(f"After updates - Columns: {df.columns.tolist()}")
            # Log first example state for verification
            if len(results) > 0:
                idx = results[0][0]
                logging.info(f"First example ({idx}) final state:")
                for col in ['Complex_CoT_orig', 'Complex_CoT_extracted', 'Complex_CoT_perturbed', 'Complex_CoT_restored', 'Complex_CoT']:
                    if col in df.columns:
                        val = df.loc[idx, col]
                        logging.info(f"- {col}: {str(val)[:100]}...")
                        
            # Clear tasks for next batch
            tasks = []
            
    return df