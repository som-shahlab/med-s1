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
    transform_to_socratic,
    transform_to_cot,
    transform_to_nejmcr_steps,
    transform_to_gemini,
    transform_to_nejmcr_transform,
    transform_to_gemini_nejmcr,
    transform_to_nejmcr_qa,
    transform_to_nejmcr_reason,
    transform_to_nejmcr_clean
)
from .clinical_formatting_ablation import transform_to_nejmcr_reason_ablated, ABLATION_CONFIGS

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
    batch_size = 350  # Increased from 25 to 100
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
        # Always pass row to transform_func since many methods need it
        tasks = [transform_func(row['Complex_CoT'], model_key, row=row, **kwargs)
                for _, row in batch.iterrows()]

        # Process batch with concurrent API calls
        batch_results = await asyncio.gather(*tasks)

        # Update results efficiently
        for (idx, row), result in zip(batch.iterrows(), batch_results):
            if result is not None:
                logging.info(f"Raw {extract_method} extraction result: {result[:100]}...")
                
                # Handle Q&A updates differently
                if extract_method == "NEJMCR Q&A":
                    # Parse Q&A result and update columns
                    lines = result.strip().split('\n')
                    for line in lines:
                        if line.startswith("Question:"):
                            df.loc[idx, 'Question'] = line[9:].strip()
                        elif line.startswith("Answer:"):
                            df.loc[idx, 'Response'] = line[7:].strip()
                    # Store raw result for debugging
                    df.loc[idx, 'Complex_CoT_extracted'] = result
                else:
                    # For other transforms, update CoT columns
                    df.loc[idx, 'Complex_CoT_extracted'] = result
                    df.loc[idx, 'Complex_CoT'] = result
                # Log updates for verification
                if extract_method == "NEJMCR Q&A":
                    logging.info(f"Updated Q&A for idx {idx}:")
                    logging.info(f"Question: {df.loc[idx, 'Question']}")
                    logging.info(f"Answer: {df.loc[idx, 'Response']}")

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
    extract_config = curation_config.get("extract")
    # Handle both string and list formats for extract
    extract_methods = [extract_config] if isinstance(extract_config, str) else (extract_config or [])
    perturbation = curation_config.get("perturbation", {})
    perturbation_type = perturbation.get("type")
    perturbation_rate = perturbation.get("rate")
    restore = curation_config.get("restore", False)

    # Load cached config
    full_config = load_full_config()

    # If perturbation is specified, we need step extraction as the final transform
    if perturbation:
        if not extract_methods or extract_methods[-1] != "step":
            logging.warning("Perturbation requires step extraction. Adding 'step' as final transform.")
            extract_methods = extract_methods + ["step"]

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
                    f"n_samples={curation_config.get('n_samples')}, extract={extract_methods}")
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
                    exp_curation_config.get('format') == curation_config.get('format') and
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
                    exp_curation_config.get('format') == curation_config.get('format') and
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
        # Apply each extraction method in sequence
        for extract_method in extract_methods:
            logging.info(f"Applying {extract_method} extraction...")
            
            # Store current state before transformation
            if 'Complex_CoT_orig' not in df.columns:
                df.loc[df['selected_for_training'], 'Complex_CoT_orig'] = df.loc[df['selected_for_training'], 'Complex_CoT']
            
            # Store pre-transform state for this method
            df.loc[df['selected_for_training'], f'Complex_CoT_before_{extract_method}'] = df.loc[df['selected_for_training'], 'Complex_CoT']
            
            # Check for ablation transforms first
            if extract_method.startswith("nejmcr-reason-"):
                ablation_name = extract_method.split("-")[-1]
                if ablation_name not in ABLATION_CONFIGS:
                    raise ValueError(f"Unknown ablation configuration: {ablation_name}")
                
                logging.info(f"Using ablation configuration: {ablation_name}")
                ablation_flags = ABLATION_CONFIGS[ablation_name]
                
                # Create wrapper for ablated transformation
                async def ablation_wrapper(cot: str, model_key: str, row: pd.Series = None, **kwargs) -> str:
                    if row is None:
                        raise ValueError("Row data is required for ablation transforms")
                    logging.info(f"Applying ablation {ablation_name} to row {row.name}")
                    return await transform_to_nejmcr_reason_ablated(
                        cot,
                        row['Question'],
                        row['Response'],
                        model_key,
                        ablation_flags
                    )
                
                df = await _apply_extraction_method(df, full_config, f"NEJMCR Reason ({ablation_name})", ablation_wrapper)
            
            elif extract_method == "step":
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
            elif extract_method in ["cot", "cot-step"]:
                # Create a wrapper function that takes Complex_CoT as input but uses Q&A
                async def cot_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    # Use Question and Response from the row directly
                    return await transform_to_cot(row['Question'], row['Response'], model_key)
                
                # Override the default filtering to use Question and Response
                df_filtered = df[df['selected_for_training'] & df['Question'].notna() & df['Response'].notna()].copy()
                df = df[~df.index.isin(df_filtered.index)].copy()  # Keep unselected rows
                
                # Add dummy Complex_CoT for _apply_extraction_method
                df_filtered['Complex_CoT'] = 'dummy'
                
                # Use existing parallelization infrastructure
                df_processed = await _apply_extraction_method(df_filtered, full_config, "Chain of Thought", cot_wrapper)
                
                # For cot-step, apply step formatting to the generated CoT
                if extract_method == "cot-step":
                    logging.info("Applying step formatting to generated CoT...")
                    # Store the CoT before step formatting
                    df_processed['Complex_CoT_orig'] = df_processed['Complex_CoT']
                    # Apply step formatting
                    df_processed = await _apply_extraction_method(df_processed, full_config, "step", transform_to_steps, extract_type="step")
                
                # Merge back
                df = pd.concat([df, df_processed])
            elif extract_method == "nejmcr":
                # Simply return Complex_CoT as-is
                logging.info("Using original Complex_CoT without transformation")
                df['Complex_CoT_extracted'] = df['Complex_CoT']
            elif extract_method == "gemini":
                # Create wrapper for Gemini transformation
                async def gemini_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    # Use Question instead of CoT for Gemini
                    return await transform_to_gemini(row['Question'], model_key, row['Response'])
                
                # Use existing parallelization infrastructure
                df = await _apply_extraction_method(df, full_config, "Gemini", gemini_wrapper)
            elif extract_method == "nejmcr-transform":
                # Create wrapper for NEJMCR transformation
                async def nejmcr_transform_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    return await transform_to_nejmcr_transform(cot, model_key, row['Response'])
                
                # Use existing parallelization infrastructure
                df = await _apply_extraction_method(df, full_config, "NEJMCR Transform", nejmcr_transform_wrapper)
            elif extract_method == "gemini-nejmcr":
                # Create wrapper for Gemini-NEJMCR transformation
                async def gemini_nejmcr_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    # Pass both Question and CoT for Gemini-NEJMCR
                    return await transform_to_gemini_nejmcr(
                        row['Question'],
                        model_key,
                        row['Response'],
                        cot  # Original CoT for enhancement
                    )
                
                # Use existing parallelization infrastructure
                df = await _apply_extraction_method(df, full_config, "Gemini-NEJMCR", gemini_nejmcr_wrapper)
            elif extract_method == "nejmcr-qa":
                # Create wrapper for NEJMCR Q&A transformation
                async def nejmcr_qa_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    # Remove "What is the diagnosis of the patient?" from question
                    original_question = row['Question']
                    clean_question = original_question.replace("\nWhat is the diagnosis of the patient?", "")
                    
                    # Get new Q&A
                    qa_result = await transform_to_nejmcr_qa(
                        clean_question,
                        cot,
                        row['Response'],
                        model_key
                    )
                    return qa_result
                
                # Use existing parallelization infrastructure
                # Q&A updates are handled in _apply_extraction_method
                df = await _apply_extraction_method(df, full_config, "NEJMCR Q&A", nejmcr_qa_wrapper)
                
            elif extract_method == "nejmcr-reason":
                # Create wrapper for standard NEJMCR reasoning transformation
                async def nejmcr_reason_wrapper(cot: str, model_key: str, row: pd.Series) -> str:
                    logging.info(f"NEJMCR Reason input for idx {row.name}:")
                    logging.info(f"Question: {row['Question']}")
                    logging.info(f"Answer: {row['Response']}")
                    logging.info(f"Original CoT: {cot[:100]}...")
                    
                    return await transform_to_nejmcr_reason(
                        cot,
                        row['Question'],
                        row['Response'],
                        model_key
                    )
                
                df = await _apply_extraction_method(df, full_config, "NEJMCR Reasoning", nejmcr_reason_wrapper)
            elif extract_method == "nejmcr-clean":
                # Create wrapper for NEJMCR clean transformation
                async def nejmcr_clean_wrapper(cot: str, model_key: str) -> str:
                    return await transform_to_nejmcr_clean(cot, model_key)
                
                # Use existing parallelization infrastructure
                df = await _apply_extraction_method(df, full_config, "NEJMCR Clean", nejmcr_clean_wrapper)
            else:
                logging.warning(f"Unknown extraction method: {extract_method}. No extraction applied.")
            
            # Store post-transform state for this method
            df.loc[df['selected_for_training'], f'Complex_CoT_after_{extract_method}'] = df.loc[df['selected_for_training'], 'Complex_CoT']

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
        logging.info(f"First example after extraction:")
        if 'Complex_CoT_orig' in first_example and pd.notna(first_example['Complex_CoT_orig']):
            logging.info(f"Original CoT length: {len(str(first_example['Complex_CoT_orig']))}")
        if 'Complex_CoT_perturbed' in first_example and pd.notna(first_example['Complex_CoT_perturbed']):
            logging.info(f"Perturbed CoT length: {len(str(first_example['Complex_CoT_perturbed']))}")
        if pd.notna(first_example['Complex_CoT']):
            logging.info(f"Final CoT length: {len(str(first_example['Complex_CoT']))}")
            logging.info(f"First 100 chars: {str(first_example['Complex_CoT'])[:100]}...")

    return df