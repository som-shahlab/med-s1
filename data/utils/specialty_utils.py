import os
import re
import logging
import pandas as pd
import random
import json
from typing import Dict, Optional, List
import asyncio
from .openai_utils import get_model_response

# Configure logging to only show warnings and errors
logging.getLogger().setLevel(logging.WARNING)

# Cache for specialties
_specialties = None
_specialties_prompt = None

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def load_specialties() -> List[str]:
    """Load medical specialties from text file with caching"""
    global _specialties
    if _specialties is None:
        with open("data/Medical_Specialties_and_Subspecialties.txt", "r") as f:
            _specialties = [line.strip() for line in f if line.strip()]
    return _specialties

def get_specialty_prompt(specialties: List[str]) -> str:
    """Get formatted specialty prompt with caching"""
    global _specialties_prompt
    if _specialties_prompt is None:
        specialties_text = "\n".join(f"{i+1:03d}. {s}" for i, s in enumerate(specialties))
        _specialties_prompt = f"""Given a medical question, determine which specialty would be primarily responsible for managing the case.

Available Specialties:
{specialties_text}

Instructions:
1. Analyze the medical concepts and procedures in the question
2. Choose the most specific applicable specialty
3. Briefly explain your choice
4. End with "SPECIALTY: XXX" where XXX is the three-digit number

Example:
Question: A patient presents with chest pain radiating to the left arm, with ST elevation on ECG.
Analysis: While Emergency Medicine might initially see this patient, Cardiology (Cardiovascular Disease) would be the primary specialist for managing acute coronary syndrome.
SPECIALTY: 030"""
    return _specialties_prompt

async def classify_specialty(
    question: str,
    specialties: List[str],
    model: str,
    max_retries: int = 3
) -> Optional[str]:
    """Classify a medical question into a specialty"""
    prompt = f"""You are a medical education expert tasked with classifying questions into medical specialties.
You must select the most appropriate specialty from the numbered list.
Your response MUST end with 'SPECIALTY: XXX' where XXX is the three-digit number from the list.

Question: {question}

{get_specialty_prompt(specialties)}

Your response:
"""
    for attempt in range(max_retries):
        try:
            response = await get_model_response(prompt, model=model)
            if not response:
                continue
                
            # Extract specialty code from response
            if match := re.search(r'SPECIALTY:\s*(\d{3})', response):
                specialty_idx = int(match.group(1)) - 1
                if 0 <= specialty_idx < len(specialties):
                    return specialties[specialty_idx]
            
            logging.warning(f"Invalid specialty code in response: {response}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = 0.1 * (2 ** attempt) * (0.5 + random.random())
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)
            else:
                logging.error(f"All attempts failed: {e}")
                break
    
    logging.warning(f"Failed to classify specialty after {max_retries} attempts")
    return None

async def batch_classify_specialties(
    questions: List[str],
    model: str,
    batch_size: Optional[int] = None
) -> List[Optional[str]]:
    """Classify a batch of questions with specialties"""
    specialties = load_specialties()
    
    # Get Gemini batch size from config if not provided
    if batch_size is None:
        config = load_config()
        batch_size = config["curation"]["gemini_batch_size"]
    
    # Process questions in true batches
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        logging.info(f"Processing specialty batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
        
        # Process batch in parallel
        tasks = [classify_specialty(q, specialties, model) for q in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        # Rate limiting between batches
        if i + batch_size < len(questions):
            await asyncio.sleep(0.1)  # 100ms between batches
            
    return results