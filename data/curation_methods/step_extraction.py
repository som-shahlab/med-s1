"""
Step extraction curation methods for the med-s1k dataset.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import asyncio
import re
from typing import Dict, Optional, List, Tuple
from utils.openai_utils import get_model_response

# Get MED_S1_DIR from environment
MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

async def transform_cot_to_steps(df: pd.DataFrame, config: Dict, extract_type: str = "step") -> pd.DataFrame:
    """
    Transform Complex_CoT column to organize it into steps or a 1-sentence summary.
    
    Args:
        df: Input dataframe with selected examples
        config: Curation configuration
        extract_type: Type of extraction to perform ("step" or "1-sentence")
        
    Returns:
        DataFrame with transformed Complex_CoT column and original preserved in Complex_CoT_orig
    """
    logging.info(f"Starting {extract_type} extraction for {len(df[df['selected_for_training']])} selected examples...")
    
    # First, add Complex_CoT_orig column to the entire dataframe
    if 'Complex_CoT_orig' not in df.columns:
        df['Complex_CoT_orig'] = df['Complex_CoT']
        logging.info("Added Complex_CoT_orig column to preserve original content")
    
    # Only process selected examples
    selected_df = df[df['selected_for_training']].copy()
    
    # Define the prompt based on extraction type
    if extract_type == "step":
        prompt_template = """
You are an expert medical educator. Your task is to transform the following chain of thought reasoning into a clear, step-by-step format.

Each step should:
1. Be numbered and have a clear title (e.g., "## Step 1: Assess the patient's condition")
2. Include all content of the original reasoning
3. Be organized in a logical sequence
4. Maintain all medical accuracy and details from the original text

Here's the chain of thought reasoning to transform:

{cot}

IMPORTANT: Your response must start directly with "## Step 1:" without any introduction or preamble. Do not include any text before the first step.
"""
    else:  # 1-sentence
        prompt_template = """
You are an expert medical educator. Your task is to transform the following chain of thought reasoning into a single, comprehensive sentence.

The sentence should:
1. Capture the key reasoning steps and logic from the original text
2. Be concise but complete, covering the main diagnostic process
3. Maintain all medical accuracy from the original text
4. Be no longer than 80 words

Here's the chain of thought reasoning to transform:

{cot}

IMPORTANT: Your response must be EXACTLY ONE SENTENCE. Do not include any introduction, explanation, or multiple sentences. Start directly with the sentence and end with a period. Do not include any text before or after the sentence.
"""
    
    # Process in batches to avoid rate limits
    batch_size = 10
    total_batches = (len(selected_df) + batch_size - 1) // batch_size
    
    for i in range(0, len(selected_df), batch_size):
        batch = selected_df.iloc[i:i+batch_size]
        batch_idx = batch.index
        
        logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} examples)")
        
        # Process each example in the batch
        tasks = []
        for idx, row in batch.iterrows():
            if pd.isna(row['Complex_CoT']) or not row['Complex_CoT'].strip():
                continue
                
            prompt = prompt_template.format(cot=row['Complex_CoT'])
            # Use "base_judge" as fallback if "curation" is not defined
            model_key = config.get("model_choices", {}).get("curation", config.get("model_choices", {}).get("base_judge", "gemini-2.0-flash"))
            
            # Check if model_key exists in config["models"]
            if model_key not in config.get("models", {}):
                logging.warning(f"Model key '{model_key}' not found in config. Using gemini-2.0-flash as fallback.")
                model_key = "gemini-2.0-flash"
                
            tasks.append(get_model_response(prompt, model=model_key, max_tokens=4096))
        
        # Wait for all tasks to complete
        batch_results = await asyncio.gather(*tasks)
        
        # Update the dataframe with the results
        result_idx = 0
        for idx, row in batch.iterrows():
            if pd.isna(row['Complex_CoT']) or not row['Complex_CoT'].strip():
                continue
                
            if batch_results[result_idx] is not None:
                # Post-process the result based on extraction type
                result = batch_results[result_idx]
                
                # Log the raw result for debugging
                logging.info(f"Raw {extract_type} extraction result: {result[:100]}...")
                
                if extract_type == "step":
                    # Check if "## Step 1:" exists in the string and clip to that point
                    step1_match = re.search(r'## Step 1:', result)
                    if step1_match:
                        result = result[step1_match.start():]
                    else:
                        logging.warning(f"Step extraction did not produce expected format. Raw result: {result[:100]}...")
                else:  # 1-sentence
                    # Just basic cleanup for the 1-sentence result
                    result = result.strip()
                    
                    # Remove any extra newlines or multiple spaces
                    result = re.sub(r'\s+', ' ', result)
                    
                    # If it's too long, truncate with ellipsis
                    if len(result) > 700:
                        result = result[:697] + "..."
                        
                    # Ensure it ends with a period
                    if not result.endswith('.'):
                        result = result + '.'
                    
                    logging.info(f"Processed 1-sentence result: {result}")
                
                # Update the Complex_CoT column with the transformed format
                df.loc[idx, 'Complex_CoT'] = result
                
                # No need to set Complex_CoT_orig again as it's already set for the entire dataframe
            
            result_idx += 1
    
    logging.info(f"Completed {extract_type} extraction for {len(selected_df)} examples")
    return df

async def apply_step_extraction(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply extraction to the selected examples based on the extract method.
    
    Args:
        df: Input dataframe with selected examples
        config: Curation configuration
        
    Returns:
        DataFrame with transformed Complex_CoT column based on extraction method
    """
    # Check extraction method
    curation_config = config.get("curation", {})
    extract_method = curation_config.get("extract")
    
    if extract_method in ["step", "1-sentence"]:
        logging.info(f"Applying {extract_method} extraction to selected examples")
        
        # Directly await the transform function since we're already in an async context
        with open(os.path.join(MED_S1_DIR, "config.json"), 'r') as config_file:
            full_config = json.load(config_file)
        
        # Pass the extract_type parameter to transform_cot_to_steps
        df = await transform_cot_to_steps(df, full_config, extract_type=extract_method)
        
        # Log the first example after transformation for debugging
        selected_df = df[df['selected_for_training']]
        if len(selected_df) > 0:
            first_example = selected_df.iloc[0]
            logging.info(f"First example after {extract_method} extraction:")
            if 'Complex_CoT_orig' in first_example and pd.notna(first_example['Complex_CoT_orig']):
                logging.info(f"Original CoT length: {len(str(first_example['Complex_CoT_orig']))}")
            if pd.notna(first_example['Complex_CoT']):
                logging.info(f"Transformed CoT length: {len(str(first_example['Complex_CoT']))}")
                logging.info(f"First 100 chars: {str(first_example['Complex_CoT'])[:100]}...")
    
    return df