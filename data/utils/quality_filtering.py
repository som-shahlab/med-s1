import pandas as pd
import logging
from datetime import datetime
from typing import Dict
import os

def get_output_dir() -> str:
    """Get the output directory from environment"""
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    if not output_dir:
        raise ValueError("MED_S1K_OUTPUT environment variable not set")
    return output_dir

def quality_filter(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Filter out empty/null values and add quality metadata"""
    logging.info(f"Starting quality filter with {len(df)} examples...")
    
    # Add quality metadata columns
    df['has_question'] = ~df['Question'].isna()
    df['has_cot'] = ~df['Complex_CoT'].isna()
    df['has_response'] = ~df['Response'].isna()
    df['quality_score'] = df[['has_question', 'has_cot', 'has_response']].sum(axis=1)
    
    # Initialize filter tracking
    df['filter_status'] = 'kept'
    df['filter_stage'] = None
    df['filter_reason'] = None
    
    # Mark quality filter status
    quality_mask = df[['Question', 'Complex_CoT', 'Response']].isna().any(axis=1)
    df.loc[quality_mask, 'filter_status'] = 'removed'
    df.loc[quality_mask, 'filter_stage'] = 'quality'
    df.loc[quality_mask, 'filter_reason'] = df[quality_mask].apply(
        lambda x: "missing_" + ",".join([
            col.lower() for col, value in zip(['Question', 'Complex_CoT', 'Response'],
                                           [x['Question'], x['Complex_CoT'], x['Response']])
            if pd.isna(value)
        ]),
        axis=1
    )
    
    # Add timestamp
    df['quality_filter_timestamp'] = datetime.now().isoformat()
    
    # Save intermediate state
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_post_quality_{timestamp}.parquet"))
    
    # Log quality filter results
    quality_filtered = df[df['filter_stage'] == 'quality']
    logging.info("=== Quality Filter Results ===")
    logging.info(f"Total examples: {len(df)}")
    logging.info(f"Kept: {len(df[df['filter_status'] == 'kept'])}")
    logging.info(f"Removed: {len(quality_filtered)}")
    logging.info("\nRemoval reasons:")
    for reason, count in quality_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    logging.info(f"\nQuality score distribution:\n{df['quality_score'].value_counts().sort_index()}")
    
    return df