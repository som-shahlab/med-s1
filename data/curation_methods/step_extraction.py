"""
Step extraction and clinical formatting methods for the med-s1k dataset.
"""

import pandas as pd
import os
import json
import logging
import asyncio
from typing import Dict
from functools import lru_cache
from .clinical_formatting import (
    transform_to_list,
    transform_to_markdown,
    transform_to_step_evidence,
    transform_to_note,
    transform_to_steps
)

# Get MED_S1_DIR from environment
MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

# Cache config loading
@lru_cache(maxsize=1)
def load_full_config():
    with open(os.path.join(MED_S1_DIR, "config.json"), 'r') as config_file:
        return json.load(config_file)

async def _apply_extraction_method(df: pd.DataFrame, config: Dict, extract_method: str, transform_func, **kwargs):
    """
    Applies a given transformation function to the selected examples in batches.

    Args:
        df: Input dataframe.
        config: Curation configuration.
        extract_method: Name of the extraction method (for logging).
        transform_func: Asynchronous function to apply for transformation.
        **kwargs: Additional keyword arguments to pass to transform_func.

    Returns:
        DataFrame with transformed Complex_CoT column.
    """
    logging.info(f"Applying {extract_method} extraction")

    # Pre-filter DataFrame and add Complex_CoT_orig column if needed
    if 'Complex_CoT_orig' not in df.columns:
        df.loc[df['selected_for_training'], 'Complex_CoT_orig'] = df.loc[df['selected_for_training'], 'Complex_CoT']
        logging.info("Added Complex_CoT_orig column to preserve original content")

    # Pre-filter to only valid rows to process
    selected_df = df[df['selected_for_training'] & df['Complex_CoT'].notna() & (df['Complex_CoT'].str.strip() != '')].copy()
    
    # Increased batch size for better throughput
    batch_size = 100  # Increased from 25 to 100
    total_batches = (len(selected_df) + batch_size - 1) // batch_size

    # Get model key once
    model_key = config.get("model_choices", {}).get("curation", 
                config.get("model_choices", {}).get("base_judge", "gemini-2.0-flash"))
    if model_key not in config.get("models", {}):
        logging.warning(f"Model key '{model_key}' not found in config. Using gemini-2.0-flash as fallback.")
        model_key = "gemini-2.0-flash"

    # Process in larger batches
    for i in range(0, len(selected_df), batch_size):
        batch = selected_df.iloc[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} examples)")

        # Create tasks for all valid rows at once
        tasks = [transform_func(row['Complex_CoT'], model_key, **kwargs) 
                for _, row in batch.iterrows()]

        # Process batch with concurrent API calls
        batch_results = await asyncio.gather(*tasks)

        # Update results efficiently
        for (idx, _), result in zip(batch.iterrows(), batch_results):
            if result is not None:
                logging.info(f"Raw {extract_method} extraction result: {result[:100]}...")
                df.loc[idx, 'Complex_CoT'] = result

    logging.info(f"Completed {extract_method} extraction for {len(selected_df)} examples")
    return df

async def apply_step_extraction(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Apply extraction and optional perturbation to the selected examples.

    Args:
        df: Input dataframe with selected examples
        config: Curation configuration

    Returns:
        DataFrame with transformed Complex_CoT column based on extraction method and perturbation
    """
    # Check extraction method and perturbation
    curation_config = config.get("curation", {})
    extract_method = curation_config.get("extract")
    perturbation = curation_config.get("perturbation", {})
    perturbation_type = perturbation.get("type")
    perturbation_rate = perturbation.get("rate")

    # Load cached config
    full_config = load_full_config()

    # If perturbation is specified, we need step extraction
    if perturbation and extract_method != "step":
        logging.warning("Perturbation requires step extraction. Forcing extract_method to 'step'")
        extract_method = "step"

    # Check if we can reuse existing step extraction
    if perturbation:
        results_json = os.environ.get('RESULTS_JSON')
        if results_json:
            with open(results_json, 'r') as f:
                results = json.load(f)
            
            # Look for matching experiment without perturbation
            for exp_name, exp_data in results['experiments'].items():
                exp_config = exp_data.get('config', {}).get('curation', {})
                if (exp_config.get('method') == curation_config.get('method') and
                    exp_config.get('n_samples') == curation_config.get('n_samples') and
                    exp_config.get('extract') == 'step' and
                    not exp_config.get('perturbation')):
                    
                    # Found matching experiment, try to load its filtered dataset
                    filtered_path = exp_data.get('paths', {}).get('filtered')
                    if filtered_path and os.path.exists(filtered_path):
                        logging.info(f"Reusing step extraction from {exp_name}")
                        df = pd.read_parquet(filtered_path)
                        break

    # Apply appropriate transformation
    if extract_method == "step":
        df = await _apply_extraction_method(df, full_config, "step", transform_to_steps, extract_type="step")
    elif extract_method == "1-sentence":
        df = await _apply_extraction_method(df, full_config, "1-sentence", transform_to_steps, extract_type="1-sentence")
    elif extract_method == "list":
        df = await _apply_extraction_method(df, full_config, "list", transform_to_list)
    elif extract_method == "markdown":
        df = await _apply_extraction_method(df, full_config, "markdown", transform_to_markdown)
    elif extract_method == "step-evidence":
        df = await _apply_extraction_method(df, full_config, "step-evidence", transform_to_step_evidence)
    elif extract_method in ["note", "soap", "soapie", "isbar", "pomr"]:
        df = await _apply_extraction_method(df, full_config, extract_method.upper(), transform_to_note)
    else:
        logging.warning(f"Unknown extraction method: {extract_method}. No extraction applied.")

    # Apply perturbation if specified
    if perturbation_type:
        from .perturbation import apply_perturbation
        
        # Get model key for LLM-based perturbations
        model_key = None
        if perturbation_type in ["collapse_consecutive", "answer"]:
            model_key = full_config.get("model_choices", {}).get("curation",
                        full_config.get("model_choices", {}).get("base_judge", "gemini-2.0-flash"))
        
        df = await apply_perturbation(df, full_config, perturbation_type, perturbation_rate, model_key)

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