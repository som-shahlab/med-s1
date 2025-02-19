import os
import sys
import time
import json
from typing import Optional, Sequence, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
import atexit
from functools import partial
from huggingface_hub import login

_model = None  # Global model instance

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("config.json", "r") as f:
        return json.load(f)

def cleanup_model():
    """Safely cleanup model resources"""
    global _model
    if _model and hasattr(_model, 'llm_engine'):
        try:
            _model.llm_engine._cleanup()
        except:
            pass
        _model = None

def initialize_model(
    model_name: str,
    tokenizer_path: str,
    max_retries: int = 3
) -> bool:
    """Initialize the global model instance"""
    global _model
    
    if _model is not None:
        return True
    
    # Set HuggingFace token from environment
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        logging.error("HUGGING_FACE_HUB_TOKEN not set")
        return False
    
    # Login to HuggingFace (only once)
    try:
        login(token=token)
    except Exception as e:
        logging.error(f"Failed to login to HuggingFace: {e}")
        return False
        
    for attempt in range(max_retries):
        try:
            _model = LLM(
                model=model_name,
                tokenizer=tokenizer_path,
                tensor_parallel_size=1,
                max_num_batched_tokens=131072,  # Doubled again for A100
                gpu_memory_utilization=0.95,    # Keep at 95%
                max_model_len=32768,
                enforce_eager=False,            # Keep CUDA graph enabled
                trust_remote_code=True,
                device="cuda",
                dtype="bfloat16"
            )
            atexit.register(cleanup_model)
            return True
        except Exception as e:
            logging.error(f"Error loading model (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return False

def _llama_forward(
    prompts: Sequence[str],
    model_name: str,
    tokenizer_path: str,
    max_length: int = 32768,
    temperature: float = 0.05,
) -> Optional[Sequence[str]]:
    """Forward pass through local Llama model using vLLM"""
    global _model
    
    # Initialize model if needed
    if not initialize_model(model_name, tokenizer_path):
        return [None] * len(prompts)
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1024,
        stop=["</s>", "\n\n"],
        top_p=0.9,
        frequency_penalty=0.1
    )
    
    try:
        # Get LLaMA batch size from config
        config = load_config()
        batch_size = config["curation"]["llama_batch_size"]
        all_outputs = []
        
        # Process prompts in batches with memory management
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            try:
                outputs = _model.generate(
                    prompts=batch_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True
                )
                all_outputs.extend([output.outputs[0].text for output in outputs])
                
                # # Force cache clearing after each batch
                # if hasattr(_model, 'llm_engine'):
                #     _model.llm_engine._cleanup()
                    
                # Small delay between batches
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Batch generation error: {e}")
                all_outputs.extend([None] * len(batch_prompts))
        
        return all_outputs
    except Exception as e:
        logging.error(f"Error in LLaMA inference: {e}")
        return [None] * len(prompts)

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
        max_length=104000  # Match max_model_len
    )