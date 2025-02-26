import asyncio
import logging
import time
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from .model_utils import get_base_model_answers
from .openai_utils import verify_answer
from .specialty_utils import batch_classify_specialties

async def verify_answers_batch(questions: List[str], answers: List[str], correct_answers: List[str]) -> List[Tuple[bool, str]]:
    """Verify a batch of answers using async calls"""
    tasks = []
    for q, a, c in zip(questions, answers, correct_answers):
        if a is not None:
            tasks.append(verify_answer(q, a, c))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task for None answers
    
    results = await asyncio.gather(*tasks)
    return [r if isinstance(r, tuple) else (False, "Failed to process") for r in results]

async def llama_worker(df: pd.DataFrame, llama_queue: asyncio.Queue, batch_size: int):
    """Process questions through Llama model in batches"""
    df_to_process = df[df['filter_status'] == 'kept'].copy()
    questions = df_to_process['Question'].tolist()
    total_processed = 0
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_answers = get_base_model_answers(batch_questions)
        
        # Put entire batch into queue
        batch_data = list(zip(batch_questions, batch_answers))
        await llama_queue.put(batch_data)
        
        total_processed += len(batch_questions)
        if total_processed % 100 == 0:
            logging.info(f"Processed {total_processed} questions through Llama")
    
    # Signal completion
    await llama_queue.put(None)

async def gemini_worker(df: pd.DataFrame, llama_queue: asyncio.Queue, specialty_queue: asyncio.Queue, batch_size: int):
    """Verify answers with Gemini using batched processing"""
    results_dict = {}  # Store results by question
    total_verified = 0
    hard_questions = []  # Collect hard questions for specialty worker
    start_time = time.time()
    
    try:
        while True:
            try:
                batch_data = await asyncio.wait_for(llama_queue.get(), timeout=0.1)
                if batch_data is None:
                    break
                
                # Unpack batch data
                batch_questions = []
                batch_answers = []
                batch_correct = []
                
                for question, answer in batch_data:
                    correct_answer = df[df['Question'] == question]['Response'].iloc[0]
                    batch_questions.append(question)
                    batch_answers.append(answer)
                    batch_correct.append(correct_answer)
                
                total_to_verify = len(batch_questions)
                verified_count = 0
                
                # Process in sub-batches to stay within API limits
                for i in range(0, len(batch_questions), batch_size):
                    sub_batch_questions = batch_questions[i:i + batch_size]
                    sub_batch_answers = batch_answers[i:i + batch_size]
                    sub_batch_correct = batch_correct[i:i + batch_size]
                    
                    results = await verify_answers_batch(
                        sub_batch_questions,
                        sub_batch_answers,
                        sub_batch_correct
                    )
                    
                    verified_count += len(sub_batch_questions)
                    
                    # Store results and collect hard questions
                    for q, (is_correct, judgment), answer in zip(sub_batch_questions, results, sub_batch_answers):
                        results_dict[q] = {
                            'base_model_response': answer,
                            'base_model_correct': is_correct,
                            'base_model_judgment': judgment
                        }
                        if not is_correct:  # Hard question
                            hard_questions.append(q)
                    
                    # Rate limiting
                    await asyncio.sleep(len(sub_batch_questions) * 60/2000)  # Stay under 2000 RPM
                
                # Send hard questions to specialty worker immediately
                for q in hard_questions:
                    await specialty_queue.put(q)
                hard_questions = []
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue
        
        # Process any remaining hard questions
        for q in hard_questions:
            await specialty_queue.put(q)
        
        # Update DataFrame with all results
        for question, result in results_dict.items():
            mask = df['Question'] == question
            df.loc[mask, 'base_model_response'] = result['base_model_response']
            df.loc[mask, 'base_model_correct'] = result['base_model_correct']
            df.loc[mask, 'base_model_judgment'] = result['base_model_judgment']
        
        # Signal completion to specialty worker
        await specialty_queue.put(None)
        logging.info("Gemini verification complete, signaled specialty worker")
        
    except Exception as e:
        logging.error(f"Fatal error in gemini_worker: {str(e)}")
        await specialty_queue.put(None)  # Signal completion even on error
        raise

async def specialty_worker(df: pd.DataFrame, specialty_queue: asyncio.Queue, config: Dict, batch_size: int):
    """Label specialties for hard questions as they come in from gemini worker"""
    questions_to_process = []
    total_processed = 0
    start_time = time.time()
    
    while True:
        try:
            # Get next batch or None if done
            batch = await asyncio.wait_for(specialty_queue.get(), timeout=0.1)
            if batch is None:
                # Process any remaining questions
                if questions_to_process:
                    specialties = await batch_classify_specialties(
                        questions=questions_to_process,
                        model=config["model_choices"]["specialty_labeler"],
                        batch_size=len(questions_to_process)
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
                
                specialties = await batch_classify_specialties(
                    questions=current_batch,
                    model=config["model_choices"]["specialty_labeler"],
                    batch_size=batch_size
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
                specialties = await batch_classify_specialties(
                    questions=questions_to_process,
                    model=config["model_choices"]["specialty_labeler"],
                    batch_size=len(questions_to_process)
                )
                # Update dataframe
                for q, s in zip(questions_to_process, specialties):
                    df.loc[df['Question'] == q, 'specialty'] = s
                total_processed += len(questions_to_process)
                questions_to_process = []
    
    # Add timestamp after completion
    df['specialty_label_timestamp'] = datetime.now().isoformat()
    logging.info(f"Completed specialty classification: {total_processed} questions processed")

async def run_pipeline(df: pd.DataFrame, config: Dict, batch_size: int):
    """Run the full pipeline with all workers"""
    # Create queues for communication between workers
    llama_queue = asyncio.Queue()
    specialty_queue = asyncio.Queue()
    
    # Run all workers concurrently
    await asyncio.gather(
        llama_worker(df, llama_queue, batch_size),
        gemini_worker(df, llama_queue, specialty_queue, batch_size),
        specialty_worker(df, specialty_queue, config, batch_size)
    )
    
    return df