import os
import sys
import time
import json
from typing import Optional, Sequence, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
from functools import partial

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def _llama_forward(
    prompts: Sequence[str],
    model_name: str,
    tokenizer_path: str,
    max_length: int = 128000,
    temperature: float = 0.05,
) -> Optional[Sequence[str]]:
    """Forward pass through local Llama model using vLLM"""
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_length
    )
    
    # Use 2 GPUs for 8B model
    tensor_parallel_size = 2
    
    model = None
    while model is None:
        try:
            model = LLM(
                model=model_name,
                tokenizer=tokenizer_path,
                tensor_parallel_size=tensor_parallel_size
            )
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            time.sleep(10)
            
    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params
    )
    
    result = []
    for output in outputs:
        result.append(output.outputs[0].text)
    return result

def format_medical_prompt(question: str) -> str:
    """Format medical question for base model"""
    return f"""You are an expert medical professional. Answer the following medical question accurately and concisely.

Question: {question}

Think through this step-by-step:
1."""

def get_base_model_answers(questions: Sequence[str]) -> Sequence[Optional[str]]:
    """Get answers from base Llama model for multiple questions"""
    config = load_config()
    model_config = config["models"][config["model_choices"]["base"]]
    
    prompts = [format_medical_prompt(q) for q in questions]
    
    return _llama_forward(
        prompts=prompts,
        model_name=model_config["hf_path"],
        tokenizer_path=model_config["hf_path"],
        max_length=model_config["max_length"]
    )