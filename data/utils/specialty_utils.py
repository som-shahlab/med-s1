import os
import re
import logging
import pandas as pd
from typing import Dict, Optional, List
import asyncio
from .openai_utils import get_model_response
def load_specialties() -> List[str]:
    """Load medical specialties from text file"""
    with open("data/Medical_Specialties_and_Subspecialties.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]

def format_specialty_prompt(specialties: List[str]) -> str:
    """Format specialties into a numbered list for classification"""
    return "\n".join(f"{i+1:03d}. {s}" for i, s in enumerate(specialties))

async def classify_specialty(
    question: str,
    specialties: List[str],
    model: str,
    max_retries: int = 3
) -> Optional[str]:
    """Classify a medical question into a specialty"""
    specialties_text = format_specialty_prompt(specialties)
    
    system_prompt = ("You are a medical education expert tasked with classifying questions into medical specialties. "
                    "You must select the most appropriate specialty from the numbered list. "
                    "Your response MUST end with 'SPECIALTY: XXX' where XXX is the three-digit number from the list.")
    
    user_prompt = f"""Given a medical question, determine which specialty would be primarily responsible for managing the case.

Question: {question}

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
SPECIALTY: 030

Your response:
"""
    for attempt in range(max_retries):
        try:
            response = await get_model_response(user_prompt, model=model)
            if not response:
                continue
                
            # Extract specialty code from response
            if match := re.search(r'SPECIALTY:\s*(\d{3})', response):
                specialty_idx = int(match.group(1)) - 1
                if 0 <= specialty_idx < len(specialties):
                    return specialties[specialty_idx]
            
            logging.warning(f"Invalid specialty code in response: {response}")
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
    
    logging.error(f"Failed to classify specialty after {max_retries} attempts")
    return None

async def batch_classify_specialties(
    questions: List[str],
    model: str,
    batch_size: int = 4
) -> List[Optional[str]]:
    """Classify a batch of questions with specialties"""
    specialties = load_specialties()
    all_specialties = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        tasks = [classify_specialty(q, specialties, model) for q in batch]
        results = await asyncio.gather(*tasks)
        all_specialties.extend(results)
        
        # Small delay between batches
        await asyncio.sleep(1)
    
    return all_specialties