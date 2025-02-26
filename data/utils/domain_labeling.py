import pandas as pd
import logging
import time
from datetime import datetime
from typing import Dict, List
import asyncio
from tqdm import tqdm
import os
from .specialty_utils import batch_classify_specialties

def batch_label_specialties(df: pd.DataFrame, config: Dict, batch_size: int = 4) -> pd.DataFrame:
    """Label questions with specialties in batches with rate limiting"""
    start_time = time.time()
    total_examples = len(df)
    logging.info(f"Starting specialty classification for {total_examples} examples")
    
    # Only process examples that passed previous filters
    df_to_process = df[df['filter_status'] == 'kept'].copy()
    questions = df_to_process['Question'].tolist()
    
    # Process in batches
    total_tokens = 0  # Initialize token counter
    all_specialties = asyncio.run(batch_classify_specialties(
        questions=questions,
        model=config["model_choices"]["specialty_labeler"],
        batch_size=batch_size
    ))
    
    # Update dataframe
    df.loc[df['filter_status'] == 'kept', 'specialty'] = all_specialties
    df['specialty_label_timestamp'] = datetime.now().isoformat()
    
    # Mark examples without labels
    specialty_mask = (df['filter_status'] == 'kept') & df['specialty'].isna()
    df.loc[specialty_mask, 'filter_status'] = 'removed'
    df.loc[specialty_mask, 'filter_stage'] = 'specialty'
    df.loc[specialty_mask, 'filter_reason'] = 'no_specialty_assigned'
    
    # Save intermediate state
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(output_dir, f"med_s1k_post_specialty_{timestamp}.parquet"))
    
    elapsed_time = time.time() - start_time
    specialty_filtered = df[df['filter_stage'] == 'specialty']
    
    logging.info("=== Specialty Classification Results ===")
    logging.info(f"Processing time: {elapsed_time:.2f}s")
    
    logging.info("\nClassification results:")
    logging.info(f"Total examples processed: {len(df_to_process)}")
    logging.info(f"Successfully classified: {len(df[df['specialty'].notna() & (df['filter_status'] == 'kept')])}")
    logging.info(f"Failed to classify: {len(specialty_filtered)}")
    
    logging.info("\nFiltering results:")
    logging.info(f"Kept from previous stage: {len(df_to_process)}")
    logging.info(f"Removed in this stage: {len(specialty_filtered)}")
    for reason, count in specialty_filtered['filter_reason'].value_counts().items():
        logging.info(f"- {reason}: {count}")
    
    logging.info("\nSpecialty distribution for kept examples:")
    for specialty, count in df[df['filter_status'] == 'kept']['specialty'].value_counts().items():
        logging.info(f"- {specialty}: {count}")
    
    return df

async def process_specialty_batch(questions: List[str], config: Dict, batch_size: int) -> List[str]:
    """Process a batch of questions for specialty classification"""
    specialties = await batch_classify_specialties(
        questions=questions,
        model=config["model_choices"]["specialty_labeler"],
        batch_size=batch_size
    )
    return specialties

async def specialty_worker(queue: asyncio.Queue, df: pd.DataFrame, config: Dict, batch_size: int):
    """Worker for processing specialty classification tasks"""
    questions_to_process = []
    total_processed = 0
    start_time = time.time()
    
    while True:
        try:
            # Get next batch or None if done
            batch = await asyncio.wait_for(queue.get(), timeout=0.1)
            if batch is None:
                # Process any remaining questions
                if questions_to_process:
                    specialties = await process_specialty_batch(
                        questions_to_process,
                        config,
                        batch_size
                    )
                    # Update dataframe
                    for q, s in zip(questions_to_process, specialties):
                        df.loc[df['Question'] == q, 'specialty'] = s
                    total_processed += len(questions_to_process)
                break
            
            # Add new questions to processing queue
            if isinstance(batch, list):
                questions_to_process.extend(batch)
            else:
                questions_to_process.append(batch)
            
            # Process in batches when we have enough questions
            while len(questions_to_process) >= batch_size:
                current_batch = questions_to_process[:batch_size]
                questions_to_process = questions_to_process[batch_size:]
                
                specialties = await process_specialty_batch(
                    current_batch,
                    config,
                    batch_size
                )
                
                # Update dataframe
                for q, s in zip(current_batch, specialties):
                    df.loc[df['Question'] == q, 'specialty'] = s
                
                total_processed += len(current_batch)
                
                # Rate limiting
                await asyncio.sleep(len(current_batch) * 60/2000)  # Stay under 2000 RPM
                
                if total_processed % 50 == 0:
                    logging.info(f"Processed {total_processed} specialties")
                    
                    # Show specialty distribution
                    current_dist = df[df['specialty'].notna()]['specialty'].value_counts()
                    logging.info("\nCurrent specialty distribution:")
                    for specialty, count in current_dist.head().items():
                        logging.info(f"- {specialty}: {count}")
        
        except asyncio.TimeoutError:
            # Process accumulated questions if we have enough
            if len(questions_to_process) >= batch_size // 2:
                specialties = await process_specialty_batch(
                    questions_to_process,
                    config,
                    len(questions_to_process)
                )
                # Update dataframe
                for q, s in zip(questions_to_process, specialties):
                    df.loc[df['Question'] == q, 'specialty'] = s
                total_processed += len(questions_to_process)
                questions_to_process = []
    
    # Add timestamp after completion
    df['specialty_label_timestamp'] = datetime.now().isoformat()
    logging.info(f"Completed specialty classification: {total_processed} questions processed")