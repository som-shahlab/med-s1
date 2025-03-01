import sglang as sgl
import asyncio
import gc
from transformers import AutoTokenizer
from jinja2 import Template
from typing import List, Dict, Optional
from tqdm.auto import tqdm

# Constants for token limits
MAX_NEW_TOKENS = 2048  # Maximum new tokens per LLM call

class TestTimeScaler:
    """Handles test time scaling evaluation with efficient KV cache reuse"""
    
    def __init__(self, engine: sgl.Engine, tokenizer: AutoTokenizer):
        """Initialize scaler with sglang engine and tokenizer
        
        Args:
            engine: sglang Engine instance for inference
            tokenizer: Tokenizer for the model
        """
        self.engine = engine
        self.tokenizer = tokenizer
        self.is_llama = "Llama" in tokenizer.name_or_path
        
        # Set up special tokens
        self.begin_text = "<|begin_of_text|>"
        self.system_start = "<|start_header_id|>system<|end_header_id|>"
        self.user_start = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
        self.think_token = "<|start_header_id|>think<|end_header_id|>"
        self.answer_token = "<|start_header_id|>answer<|end_header_id|>"
        
        # Define approaches
        self.approaches = {
            "immediate": self.immediate_answer,
            "reasoning": self.normal_reasoning,
            "reasoning_2x": lambda x, temp, debug: self.extended_reasoning(x, 2, temp, debug),
            "reasoning_4x": lambda x, temp, debug: self.extended_reasoning(x, 4, temp, debug)
        }

    def format_prompt(self, question: str, assistant_content: str = "") -> str:
        """Format prompt manually following training format"""
        system_msg = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
        # Manually construct prompt without eot_id tokens
        return f"{self.begin_text}{self.system_start}\n{system_msg}\n{self.user_start}\n{question}\n{self.assistant_start}\n{assistant_content}"

    async def immediate_answer(self, prompt: str, temperature: float = 0, debug: bool = False) -> Dict:
        """Get immediate answer without additional reasoning"""
        if debug:
            print("\n" + "="*80)
            print("IMMEDIATE ANSWER APPROACH")
            print("="*80)
            print("\nStage: Preparing prompt")
            print("Action: Adding think and answer tokens to get direct answer")
            
        # Format with assistant content including think and answer tokens
        modified_prompt = self.format_prompt(prompt, f"{self.think_token}\nI have the answer.\n{self.answer_token}\n")
        
        if debug:
            print("\nInput to LLM:")
            print("-" * 40)
            print(modified_prompt)
            print("-" * 40)
            print("\nStage: Generating answer")
            print("Action: Single LLM call to get answer (no reasoning)")
            
        # Generate answer
        output = await self.engine.async_generate(
            prompt=modified_prompt,
            sampling_params={
                "stop": ["<|eot_id|>"],
                "temperature": temperature,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        )
        
        if debug:
            print("\nLLM Output:")
            print("-" * 40)
            print(output['text'])
            print("-" * 40)
            print("\nStage: Complete")
            print("Token count: 0 (immediate answer has no reasoning tokens)")
            
        return {
            "approach": "immediate",
            "output": modified_prompt + output['text'],
            "n_reasoning_tokens": 0  # No reasoning tokens for immediate answer
        }
    
    async def normal_reasoning(self, prompt: str, temperature: float = 0, debug: bool = False) -> Dict:
        """Normal reasoning approach with single LLM call"""
        if debug:
            print("\n" + "="*80)
            print("NORMAL REASONING APPROACH")
            print("="*80)
            print("\nStage: Preparing prompt")
            print("Action: Adding think and answer tokens for single-pass reasoning")
            
        # Format with assistant content including think and answer tokens
        current_prompt = self.format_prompt(prompt, f"{self.think_token}\n")
        
        if debug:
            print("\nLLM Call - Generate Reasoning and Answer")
            print("Input:")
            print("-" * 40)
            print(current_prompt)
            print("-" * 40)
            
        # Generate reasoning and answer in one call
        output = await self.engine.async_generate(
            prompt=current_prompt,
            sampling_params={
                "stop": ["<|eot_id|>"],
                "temperature": temperature,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        )
        
        # Extract reasoning tokens (between think and answer tokens)
        full_output = output['text']
        think_idx = full_output.find(self.think_token)
        answer_idx = full_output.find(self.answer_token)
        if think_idx != -1 and answer_idx != -1:
            reasoning = full_output[think_idx + len(self.think_token):answer_idx].strip()
            n_tokens = len(self.tokenizer.encode(reasoning))
        else:
            n_tokens = len(self.tokenizer.encode(full_output))
        
        if debug:
            print("\nOutput:")
            print("-" * 40)
            print(full_output)
            print("-" * 40)
            print("\nStage: Complete")
            print(f"Reasoning tokens: {n_tokens}")
            
        return {
            "approach": "reasoning",
            "output": current_prompt + full_output,
            "n_reasoning_tokens": n_tokens
        }
    
    async def extended_reasoning(self, prompt: str, n_iterations: int, temperature: float = 0, debug: bool = False) -> Dict:
        """Extended reasoning with multiple thinking steps"""
        approach_name = f"reasoning_{n_iterations}x"
        
        if debug:
            print("\n" + "="*80)
            print(f"EXTENDED REASONING APPROACH ({n_iterations}x)")
            print("="*80)
            print("\nStage 1: Initial Thinking")
            print("Action: Adding think token and generating initial reasoning")
            
        # Initial thinking
        current_prompt = self.format_prompt(prompt, f"{self.think_token}\n")
        
        if debug:
            print("\nLLM Call 1 - Generate Initial Reasoning")
            print("Input:")
            print("-" * 40)
            print(current_prompt)
            print("-" * 40)
            
        # Generate initial reasoning
        output = await self.engine.async_generate(
            prompt=current_prompt,
            sampling_params={
                "stop": ["<|start_header_id|>", "<|eot_id|>"],  # Allow both stop conditions
                "temperature": temperature,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        )
        reasoning = output['text']
        current_prompt += reasoning
        total_tokens = len(self.tokenizer.encode(reasoning))
        
        if debug:
            print("\nInitial Reasoning Output:")
            print("-" * 40)
            print(reasoning)
            print("-" * 40)
            print(f"Initial reasoning tokens: {total_tokens}")
        
        # Additional thinking steps
        for i in range(n_iterations):
            if debug:
                print(f"\nStage {i+2}: Extended Thinking ({i+1}/{n_iterations})")
                print("Action: Adding think token and Wait for additional reasoning")
                
            # Add think token and Wait
            current_prompt += f"{self.think_token}Wait"
            
            if debug:
                print(f"\nLLM Call {i+2} - Generate Extended Reasoning")
                print("Input:")
                print("-" * 40)
                print(current_prompt)
                print("-" * 40)
                
            # Use both start_header_id and eot_id as stop tokens
            output = await self.engine.async_generate(
                prompt=current_prompt,
                sampling_params={
                    "stop": ["<|start_header_id|>", "<|eot_id|>"],  # Allow both stop conditions
                    "temperature": temperature,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "min_new_tokens": 1
                }
            )
            additional_reasoning = output['text']
            current_prompt += additional_reasoning
            new_tokens = len(self.tokenizer.encode(additional_reasoning))
            total_tokens += new_tokens
            
            if debug:
                print("\nExtended Reasoning Output:")
                print("-" * 40)
                print(additional_reasoning)
                print("-" * 40)
                print(f"New tokens this iteration: {new_tokens}")
                print(f"Total tokens so far: {total_tokens}")
        
        if debug:
            print("\nStage Final: Generate Answer")
            print("Action: Adding answer token and generating final answer")
            
        # Get final answer
        # Keep only first think token when forcing answer
        first_think_idx = current_prompt.find(self.think_token)
        if first_think_idx != -1:
            current_prompt = current_prompt[:first_think_idx + len(self.think_token)] + current_prompt[first_think_idx + len(self.think_token):].replace(self.think_token, "")
        current_prompt += self.answer_token
        
        if debug:
            print("\nLLM Call Final - Generate Answer")
            print("Input:")
            print("-" * 40)
            print(current_prompt)
            print("-" * 40)
            
        output = await self.engine.async_generate(
            prompt=current_prompt,
            sampling_params={
                "stop": ["<|eot_id|>"],
                "temperature": temperature,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        )
        answer = output['text']
        
        if debug:
            print("\nFinal Answer Output:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            print("\nStage: Complete")
            print(f"Total reasoning tokens: {total_tokens}")
        
        return {
            "approach": approach_name,
            "output": current_prompt + answer,
            "n_reasoning_tokens": total_tokens
        }

async def evaluate_test_time_scaling(
    engine: sgl.Engine,
    tokenizer: AutoTokenizer,
    input_data: List[Dict],
    template: Optional[Template] = None,
    temperature: float = 0,
    debug: bool = False,
    debug_samples: int = 10,
    batch_size: int = 32  # Batch size for parallel processing
) -> List[Dict]:
    """Run test time scaling evaluation on input data using sglang for efficient batching
    
    Args:
        engine: sglang Engine instance for inference
        tokenizer: Tokenizer for the model
        input_data: List of evaluation examples
        template: Optional chat template (not used, we format manually)
        temperature: Sampling temperature
        debug: Whether to run in debug mode
        debug_samples: Number of samples to process in debug mode
        batch_size: Number of examples to process in parallel
        
    Returns:
        List of results with scaling approaches for each example
    """
    scaler = TestTimeScaler(engine, tokenizer)
    
    if debug:
        input_data = input_data[:debug_samples]
    
    # Format all inputs upfront
    for item in input_data:
        item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
        item["input_str"] = "Please answer the following multiple-choice question:\n{question}\n{option_str}".format_map(item)
        item["scaling_results"] = []  # Initialize scaling_results upfront
    
    # Use slightly smaller batch size to avoid OOM
    batch_size = min(batch_size, 24)  # Cap at 24 for memory safety
    
    # Calculate total steps (samples * approaches)
    total_steps = len(input_data) * len(scaler.approaches)
    final_results = []
    token_stats = {name: [] for name in scaler.approaches.keys()}
    
    # Process in batches, showing progress across all samples and approaches
    with tqdm(total=total_steps, desc="Test time scaling evaluation") as pbar:
        for batch_start in range(0, len(input_data), batch_size):
            batch_end = min(batch_start + batch_size, len(input_data))
            batch = input_data[batch_start:batch_end]
            
            # Format batch inputs
            for item in batch:
                item["scaling_results"] = []
            
            # Process each approach
            for approach_name, approach_fn in scaler.approaches.items():
                if debug:
                    print(f"\nProcessing batch {batch_start//batch_size + 1}, approach: {approach_name}")
                
                # Run approach on batch
                batch_tasks = []
                for item in batch:
                    task = approach_fn(
                        item["input_str"],
                        temperature,
                        debug=debug and batch_start == 0  # Only debug first batch
                    )
                    batch_tasks.append(task)
                
                # Wait for all tasks in batch
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Store results and update stats
                for item, result in zip(batch, batch_results):
                    item["scaling_results"].append(result)
                    token_stats[approach_name].append(result["n_reasoning_tokens"])
                    
                    # Update progress
                    desc = f"Batch {batch_start//batch_size + 1}, {approach_name} - Avg tokens: " + ", ".join(
                        f"{name}: {sum(tokens)/len(tokens):.0f}"
                        for name, tokens in token_stats.items()
                        if tokens
                    )
                    pbar.set_description(desc)
                    pbar.update(1)
            
            # Add completed batch to results
            final_results.extend(batch)
            
            # Force garbage collection between batches
            gc.collect()
            
            # Print debug info for first batch
            if debug and batch_start == 0:
                for item in batch[:debug_samples]:
                    print(f"\n{'='*80}")
                    print("Input:")
                    print("-" * 40)
                    print(item['input_str'])
                    
                    for result in item["scaling_results"]:
                        print(f"\n{'-'*40}")
                        print(f"Approach: {result['approach']}")
                        print(f"Reasoning tokens: {result['n_reasoning_tokens']}")
                        print("Output:")
                        print(result['output'])
    
    return final_results