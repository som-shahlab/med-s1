import json
import openai
import os
import pandas as pd
from typing import Dict, Optional
import logging
import time
from functools import partial

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def get_openai_response(
    prompt: str,
    model: str = "gpt4o-mini",
    max_retries: int = 3,
    retry_delay: int = 60,
) -> Optional[str]:
    """Get response from OpenAI API with retries"""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent outputs
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to get OpenAI response after {max_retries} attempts")
                return None

def verify_answer(question: str, model_answer: str, correct_answer: str) -> bool:
    """Use GPT-4-mini to verify if model's answer matches the correct answer"""
    prompt = f"""You are an expert medical knowledge evaluator. Compare the following model answer to the correct answer and determine if they are equivalent in meaning. Consider medical terminology and concepts carefully.

Question: {question}

Model's Answer: {model_answer}

Correct Answer: {correct_answer}

Are these answers equivalent in meaning? Explain your reasoning and end with a single word 'Yes' or 'No'.
"""
    response = get_openai_response(prompt)
    if response is None:
        return False
    
    # Extract final Yes/No
    return response.strip().split()[-1].lower() == "yes"

def label_specialty(question: str, specialties_df) -> Optional[str]:
    """Use GPT-4-mini to label question with medical specialty"""
    # Create a formatted list of specialties
    specialties = []
    for _, row in specialties_df.iterrows():
        if pd.notna(row['Subspecialty']) and row['Subspecialty'] != 'No Subspecialties':
            specialties.append(f"{row['Specialty']} - {row['Subspecialty']}")
        else:
            specialties.append(row['Specialty'])
    
    specialties_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(specialties))
    
    prompt = f"""You are an expert in medical education. Given a medical question, determine which medical specialty or subspecialty it most closely relates to. Choose from the following list:

{specialties_text}

Question: {question}

Analyze the question and select the most relevant specialty or subspecialty. Explain your reasoning and end with the number corresponding to your choice.
"""
    response = get_openai_response(prompt)
    if response is None:
        return None
        
    # Extract final number
    try:
        specialty_idx = int(response.strip().split()[-1]) - 1
        return specialties[specialty_idx]
    except:
        logging.error(f"Failed to parse specialty from response: {response}")
        return None