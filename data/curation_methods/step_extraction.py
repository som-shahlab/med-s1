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
    transform_to_steps,
    transform_to_qa,
    transform_to_decision_tree,
    transform_to_socratic
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
    batch_size = 500  # Increased from 25 to 100
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
                # Store extracted version in extraction column
                df.loc[idx, 'Complex_CoT_extracted'] = result
                # Always update the main column with the extraction
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
    # Get extraction parameters from config
    curation_config = config.get("curation", {})
    extract_method = curation_config.get("extract")
    perturbation = curation_config.get("perturbation", {})
    perturbation_type = perturbation.get("type")
    perturbation_rate = perturbation.get("rate")
    restore = curation_config.get("restore", False)

    # Load cached config
    full_config = load_full_config()

    # If perturbation is specified, we need step extraction
    if perturbation and extract_method != "step":
        logging.warning("Perturbation requires step extraction. Forcing extract_method to 'step'")
        extract_method = "step"

    # Initialize tracking variables
    reused_from = None
    reused_exp_has_perturbation = False
    reused_exp_has_restore = False
    
    # Track what we need to apply
    need_extraction = True
    need_perturbation = perturbation_type is not None
    need_restore = restore

    # Try to reuse existing results if possible
    results_json = os.environ.get('RESULTS_JSON')
    if results_json:
        with open(results_json, 'r') as f:
            results = json.load(f)
            
        logging.info(f"Looking for experiments to reuse with method={curation_config.get('method')}, "
                    f"n_samples={curation_config.get('n_samples')}, extract={extract_method}")
        if restore:
            logging.info(f"Need experiment with perturbation type={perturbation_type} rate={perturbation_rate} but not restored")
        elif perturbation:
            logging.info(f"Need experiment with same config but no perturbation")

        # For restore=True, look for matching experiment with same extraction and perturbation but not restored
        if restore:
            for exp_name, exp_data in results.get('experiments', {}).items():
                if not exp_data:
                    continue
                exp_curation_config = exp_data.get('config', {}).get('curation', {})
                if not exp_curation_config:
                    continue
                
                # Match everything except restore flag
                if (exp_curation_config.get('method') == curation_config.get('method') and
                    exp_curation_config.get('n_samples') == curation_config.get('n_samples') and
                    exp_curation_config.get('extract') == extract_method and
                    exp_curation_config.get('huatuo_format') == curation_config.get('huatuo_format') and
                    exp_curation_config.get('perturbation', {}).get('type') == perturbation_type and
                    exp_curation_config.get('perturbation', {}).get('rate') == perturbation_rate and
                    not exp_curation_config.get('restore')):
                    
                    exp_dir = os.path.dirname(exp_data.get('results', {}).get('curation', {}).get('dataset_path', ''))
                    if not exp_dir:
                        continue
                    filtered_path = os.path.join(exp_dir, "med_s1k_filtered.parquet")
                    if os.path.exists(filtered_path):
                        logging.info(f"Found matching non-restored experiment {exp_name}")
                        logging.info(f"Reading filtered dataset from {filtered_path}")
                        reused_df = pd.read_parquet(filtered_path)
                        
                        # First ensure we have the original content
                        if 'Complex_CoT_orig' not in df.columns:
                            df['Complex_CoT_orig'] = reused_df['Complex_CoT']
                            logging.info("Preserved original content in Complex_CoT_orig")

                        # Copy extracted version if it exists
                        if 'Complex_CoT_extracted' in reused_df.columns:
                            df['Complex_CoT_extracted'] = reused_df['Complex_CoT_extracted']
                            logging.info("Copied extracted version from reused experiment")

                        # Handle perturbed version
                        if 'Complex_CoT_perturbed' in reused_df.columns:
                            df['Complex_CoT_perturbed'] = reused_df['Complex_CoT_perturbed']
                            logging.info("Copied perturbed version from reused experiment")
                        elif need_perturbation:
                            # If we need perturbation but don't have perturbed version,
                            # we'll use the extracted version as input for perturbation
                            if 'Complex_CoT_extracted' in reused_df.columns:
                                df['Complex_CoT'] = reused_df['Complex_CoT_extracted']
                                logging.info("Using extracted version for perturbation input")
                        
                        # Set final Complex_CoT based on experiment type
                        if need_restore:
                            # For restore, we'll use the perturbed version as input
                            df['Complex_CoT'] = reused_df['Complex_CoT_perturbed']
                            logging.info("Using perturbed version for restore input")
                        elif need_perturbation:
                            # For perturbation, we'll use extracted version as input
                            if 'Complex_CoT_extracted' in reused_df.columns:
                                df['Complex_CoT'] = reused_df['Complex_CoT_extracted']
                                logging.info("Using extracted version for perturbation input")
                        else:
                            # For extraction only, use the extracted version
                            if 'Complex_CoT_extracted' in reused_df.columns:
                                df['Complex_CoT'] = reused_df['Complex_CoT_extracted']
                                logging.info("Using extracted version as final result")
                        
                        reused_from = exp_name
                        reused_exp_has_perturbation = True
                        reused_exp_has_restore = False
                        break
            
            if not reused_from:
                logging.info("No matching experiment with perturbation found for restore, will need to run full pipeline")

        # For perturbation, look for matching experiment with same extraction but no perturbation
        elif perturbation:
            logging.info("Looking for matching experiment with extraction but no perturbation")
            for exp_name, exp_data in results.get('experiments', {}).items():
                if not exp_data:
                    continue
                exp_curation_config = exp_data.get('config', {}).get('curation', {})
                if not exp_curation_config:
                    continue
                
                # Match everything except perturbation
                if (exp_curation_config.get('method') == curation_config.get('method') and
                    exp_curation_config.get('n_samples') == curation_config.get('n_samples') and
                    exp_curation_config.get('extract') == extract_method and
                    exp_curation_config.get('huatuo_format') == curation_config.get('huatuo_format') and
                    not exp_curation_config.get('perturbation')):
                    
                    exp_dir = os.path.dirname(exp_data.get('results', {}).get('curation', {}).get('dataset_path', ''))
                    if not exp_dir:
                        continue
                    filtered_path = os.path.join(exp_dir, "med_s1k_filtered.parquet")
                    if os.path.exists(filtered_path):
                        logging.info(f"Found matching non-perturbed experiment {exp_name}")
                        logging.info(f"Reading filtered dataset from {filtered_path}")
                        reused_df = pd.read_parquet(filtered_path)
                        
                        # Copy columns properly like we do for restore
                        if 'Complex_CoT_orig' not in df.columns:
                            df['Complex_CoT_orig'] = reused_df['Complex_CoT']  # Get true original
                            logging.info("Preserved original content in Complex_CoT_orig")
                        
                        # Track what we need to apply first
                        need_extraction = True
                        need_perturbation = perturbation_type is not None
                        need_restore = restore
                        
                        # Copy extracted version if it exists
                        if 'Complex_CoT_extracted' in reused_df.columns:
                            df['Complex_CoT_extracted'] = reused_df['Complex_CoT_extracted']
                            df['Complex_CoT'] = reused_df['Complex_CoT_extracted']  # Use as input
                            logging.info("Using extracted version for perturbation input")
                        else:
                            df['Complex_CoT'] = reused_df['Complex_CoT']  # Fallback to original
                            logging.info("No extracted version found, using original for perturbation input")
                        
                        reused_from = exp_name
                        reused_exp_has_perturbation = False
                        reused_exp_has_restore = False
                        break
            
            if not reused_from:
                logging.info("No matching experiment found for extraction, will need to run full pipeline")

    # Update based on what we reused
    if reused_from:
        logging.info(f"Reusing results from {reused_from}")
        need_extraction = False
        if perturbation and not reused_exp_has_perturbation:
            logging.info("Will apply perturbation to reused results")
            need_perturbation = True
        elif restore and not reused_exp_has_restore:
            logging.info("Will apply restore to reused results")
            need_restore = True
        else:
            need_perturbation = False
            need_restore = False
    else:
        logging.info("No results to reuse, will run full pipeline:")
        if need_extraction:
            logging.info("- Will run extraction")
        if need_perturbation:
            logging.info("- Will run perturbation")
        if need_restore:
            logging.info("- Will run restore")

    # Apply extraction if needed
    if need_extraction:
        logging.info("Applying extraction...")
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
        elif extract_method == "decision-tree":
            df = await _apply_extraction_method(df, full_config, "decision tree", transform_to_decision_tree)
        elif extract_method == "qa":
            df = await _apply_extraction_method(df, full_config, "Q&A", transform_to_qa)
        elif extract_method == "socratic":
            df = await _apply_extraction_method(df, full_config, "Socratic dialogue", transform_to_socratic)
        else:
            logging.warning(f"Unknown extraction method: {extract_method}. No extraction applied.")

    # Apply perturbation if needed
    if need_perturbation:
        logging.info("Applying perturbation...")
        from .perturbation import apply_perturbation
        
        # Get model key for LLM-based perturbations
        model_key = None
        if perturbation_type in ["collapse_consecutive", "answer"] or restore:
            model_key = full_config.get("model_choices", {}).get("curation",
                        full_config.get("model_choices", {}).get("base_judge", "gemini-2.0-flash"))
        
        df = await apply_perturbation(df, full_config, perturbation_type, perturbation_rate, model_key, need_restore)

    # Log the first example after transformation for debugging
    selected_df = df[df['selected_for_training']]
    if len(selected_df) > 0:
        first_example = selected_df.iloc[0]
        logging.info(f"First example after {extract_method} extraction:")
        if 'Complex_CoT_orig' in first_example and pd.notna(first_example['Complex_CoT_orig']):
            logging.info(f"Original CoT length: {len(str(first_example['Complex_CoT_orig']))}")
        if 'Complex_CoT_perturbed' in first_example and pd.notna(first_example['Complex_CoT_perturbed']):
            logging.info(f"Perturbed CoT length: {len(str(first_example['Complex_CoT_perturbed']))}")
        if pd.notna(first_example['Complex_CoT']):
            logging.info(f"Final CoT length: {len(str(first_example['Complex_CoT']))}")
            logging.info(f"First 100 chars: {str(first_example['Complex_CoT'])[:100]}...")

    return df