import json
import os
import pandas as pd
from typing import Dict, Optional, AsyncGenerator
import logging
import time
import random
import asyncio
from functools import partial
from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types

# Configure logging to only show warnings and errors
logging.getLogger().setLevel(logging.WARNING)

# Global clients for reuse
_openai_client = None
_gemini_client = None

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def get_client(model: str):
    """Get or create API client"""
    global _openai_client, _gemini_client
    config = load_config()
    model_config = config["models"][model]
    
    if model == "gpt4o-mini":
        if _openai_client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            _openai_client = AsyncOpenAI(api_key=api_key)
        return _openai_client
    else:  # Gemini models
        if _gemini_client is None:
            # Use application default credentials
            _gemini_client = genai.Client(
                vertexai=True,
                project="som-nero-phi-nigam-starr",
                location="us-central1"
            )
        return _gemini_client

async def get_model_response(
    prompt: str,
    model: str,
    max_retries: int = 3,
    initial_retry_delay: float = 0.1,  # Start with 100ms delay
    max_tokens=500
) -> Optional[str]:
    """Get response from model API with exponential backoff"""
    client = get_client(model)
    
    for attempt in range(max_retries):
        try:
            if model == "gpt4o-mini":
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:  # Gemini models
                contents = [
                    types.Content(
                        role="user",
                        parts=[{"text": prompt}]
                    )
                ]
                
                generate_config = types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.95,
                    max_output_tokens=max_tokens,
                    response_modalities=["TEXT"],
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="OFF"
                        )
                    ]
                )
                
                # Collect streaming response
                response_text = ""
                for chunk in client.models.generate_content_stream(
                    model="gemini-2.0-flash-001",
                    contents=contents,
                    config=generate_config
                ):
                    if chunk.text:
                        response_text += chunk.text
                return response_text

        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = initial_retry_delay * (2 ** attempt) * (0.5 + random.random())
                logging.warning(f"API error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(delay)
            else:
                logging.error(f"Failed to get response after {max_retries} attempts")
                return None

async def verify_answer(question: str, model_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Use model to verify if model's answer matches the correct answer and provide reasoning"""
    prompt = f"""You are an expert medical knowledge evaluator. Compare the following model answer to the correct answer and determine if they are equivalent in meaning. Consider medical terminology and concepts carefully.

Question: {question}

Model's Answer: {model_answer}

Correct Answer: {correct_answer}

Are these answers equivalent in meaning? Explain your reasoning and end with a single word 'Yes' or 'No'.
"""
    model = load_config()["model_choices"]["base_judge"]
    response = await get_model_response(prompt, model=model)
    if response is None:
        return False, "Failed to get model response"
    
    # Extract final Yes/No and keep reasoning
    parts = response.strip().split()
    is_correct = parts[-1].lower() == "yes"
    reasoning = " ".join(parts[:-1])  # Everything except the final Yes/No
    
    return is_correct, reasoning

async def label_specialty(question: str, specialties_df) -> Optional[str]:
    """Use model to label question with medical specialty"""
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
    model = load_config()["model_choices"]["specialty_labeler"]
    response = await get_model_response(prompt, model=model)
    if response is None:
        return None
        
    # Extract final number with retry logic and better parsing
    def extract_specialty_number(text: str) -> Optional[int]:
        # Try different patterns
        patterns = [
            r'(\d+)\s*$',  # Number at the end
            r'answer is (\d+)',  # "answer is X"
            r'Therefore.*?(\d+)',  # "Therefore... X"
            r'specialty.*?(\d+)',  # "specialty... X"
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                try:
                    return int(match.group(1)) - 1
                except ValueError:
                    continue
        return None

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if specialty_idx := extract_specialty_number(response):
                if 0 <= specialty_idx < len(specialties):
                    return specialties[specialty_idx]
            
            if attempt < max_retries:
                # Retry with more explicit prompt
                response = await get_model_response(
                    prompt + "\n\nIMPORTANT: Your response MUST end with a single number corresponding to your choice.",
                    model=model
                )
                if not response:
                    break
            
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                continue
            else:
                logging.error(f"All attempts failed: {e}")
                break
    
    logging.warning(f"Could not parse specialty from response: {response[:100]}...")
    return None