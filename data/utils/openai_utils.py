"""
OpenAI and Gemini API utilities with caching and retry logic.
"""

import json
import os
import pandas as pd
from typing import Dict, Optional, AsyncGenerator
import logging
import time
import random
import asyncio
import re
from functools import partial, lru_cache
from openai import OpenAI, AsyncOpenAI
from google import genai
from google.genai import types
# Configure logging to show info and above
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

# Global clients for reuse
_openai_client = None
_gemini_client = None
_config = None

MED_S1_DIR = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')

@lru_cache(maxsize=1)
def load_config() -> Dict:
    """Load configuration from config.json"""
    global _config
    if _config is None:
        with open(os.path.join(MED_S1_DIR, "config.json"), "r") as f:
            _config = json.load(f)
    return _config

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

# Cache for model responses
response_cache = {}
async def get_model_response(
    prompt: str,
    model: str,
    max_retries: int = 15,  # Increased from 5 to 10
    initial_retry_delay: float = 1.0,
    max_tokens=8192,  # Increased from 500 to 8192 for long responses
    raise_on_failure: bool = True  # New parameter to control error handling
) -> Optional[str]:
    """Get response from model API with exponential backoff and caching.
    
    Args:
        prompt: The prompt to send to the model
        model: The model to use
        max_retries: Maximum number of retries (default: 10)
        initial_retry_delay: Initial delay between retries in seconds (default: 1.0)
        max_tokens: Maximum tokens in response (default: 8192)
        raise_on_failure: Whether to raise an exception on failure after max retries (default: True)
    
    Returns:
        The model's response text, or None if raise_on_failure is False and all retries failed
        
    Raises:
        RuntimeError: If raise_on_failure is True and all retries failed
    """
    """Get response from model API with exponential backoff and caching"""
    # Check cache first
    cache_key = f"{model}:{prompt}"
    if cache_key in response_cache:
        return response_cache[cache_key]

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
                result = response.choices[0].message.content
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
                
                # Use async generate_content for complete response
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_config
                )
                
                # Check for various failure modes
                if not response:
                    logging.error("Gemini API returned None response object")
                    raise RuntimeError("Gemini API returned None response object")
                    
                if not hasattr(response, 'text'):
                    logging.error(f"Gemini response missing text attribute: {response}")
                    raise RuntimeError("Gemini response missing text attribute")
                
                result = response.text
                
                # Treat None response as a failure that should trigger retry
                if result is None:
                    logging.error(f"Gemini API returned None response. Response object: {response}")
                    raise RuntimeError("Gemini API returned None response")
                
                if result is None:
                    logging.error(f"Gemini API returned None response. Response object: {response}")

            # Cache successful response
            response_cache[cache_key] = result
            return result

        except Exception as e:
            # Log the error with more details
            error_msg = f"API error (attempt {attempt + 1}/{max_retries}): {str(e)}"
            if hasattr(e, 'response'):
                error_msg += f"\nResponse: {e.response}"
            logging.warning(error_msg)
            
            if attempt < max_retries - 1:
                # Calculate delay with jitter
                delay = initial_retry_delay * (2 ** attempt) * (0.5 + random.random())
                logging.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                error_msg = f"Failed to get response after {max_retries} attempts"
                logging.error(error_msg)
                if raise_on_failure:
                    raise RuntimeError(error_msg) from e
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

# Pre-compile regex patterns
SPECIALTY_NUMBER_PATTERNS = [
    re.compile(r'(\d+)\s*$'),  # Number at the end
    re.compile(r'answer is (\d+)'),  # "answer is X"
    re.compile(r'Therefore.*?(\d+)'),  # "Therefore... X"
    re.compile(r'specialty.*?(\d+)')  # "specialty... X"
]

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
        
    def extract_specialty_number(text: str) -> Optional[int]:
        for pattern in SPECIALTY_NUMBER_PATTERNS:
            if match := pattern.search(text, re.IGNORECASE):
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