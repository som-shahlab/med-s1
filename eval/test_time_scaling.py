import sglang as sgl
import asyncio
import gc
from transformers import AutoTokenizer
from jinja2 import Template
from typing import List, Dict, Optional
from tqdm.auto import tqdm

# Constants for token limits
MAX_NEW_TOKENS = 2048  # Maximum new tokens per LLM call

import random

class TestTimeScaler:
    """Handles test time scaling evaluation with efficient KV cache reuse"""
    
    def __init__(self, engine: sgl.Engine, tokenizer: AutoTokenizer, reasoning_approaches=None):
        """Initialize scaler with sglang engine and tokenizer
        
        Args:
            engine: sglang Engine instance for inference
            tokenizer: Tokenizer for the model
            reasoning_approaches: Optional list of reasoning approaches to use
        """
        self.engine = engine
        self.tokenizer = tokenizer
        self.is_llama = "Llama" in tokenizer.name_or_path
        self.is_huatuo = "huatuo" in tokenizer.name_or_path.lower()
        
        # Set up special tokens
        self.begin_text = "<|begin_of_text|>"
        self.system_start = "<|start_header_id|>system<|end_header_id|>"
        self.user_start = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
        
        # Set think and answer tokens based on model type
        if self.is_huatuo:
            # HuatuoGPT uses different special tokens for thinking and answering
            # but still uses the same formatting as Llama
            self.think_token = "## Thinking"
            self.answer_token = "## Final Response"
            # Note: These are multi-token sequences, not single tokens
        else:
            self.think_token = "<|start_header_id|>think<|end_header_id|>"
            self.answer_token = "<|start_header_id|>answer<|end_header_id|>"
        
        # Define base approaches
        base_approaches = {
            "immediate": self.immediate_answer,
            "reasoning": self.normal_reasoning,
            "reasoning_1x": lambda x, temp, debug: self.extended_reasoning(x, 1, temp, debug, use_random=False),
            "reasoning_2x": lambda x, temp, debug: self.extended_reasoning(x, 2, temp, debug, use_random=False),
            "reasoning_3x": lambda x, temp, debug: self.extended_reasoning(x, 3, temp, debug, use_random=False),
            "reasoning_4x": lambda x, temp, debug: self.extended_reasoning(x, 4, temp, debug, use_random=False),
            "reasoning_1x_random": lambda x, temp, debug: self.extended_reasoning(x, 1, temp, debug, use_random=True),
            "reasoning_2x_random": lambda x, temp, debug: self.extended_reasoning(x, 2, temp, debug, use_random=True),
            "reasoning_3x_random": lambda x, temp, debug: self.extended_reasoning(x, 3, temp, debug, use_random=True),
            "reasoning_4x_random": lambda x, temp, debug: self.extended_reasoning(x, 4, temp, debug, use_random=True),
            "reasoning_6x_random": lambda x, temp, debug: self.extended_reasoning(x, 6, temp, debug, use_random=True)
        }
        
        # Use specified approaches if provided, otherwise use default set
        if reasoning_approaches:
            self.approaches = {name: base_approaches[name] for name in reasoning_approaches if name in base_approaches}
        else:
            # Default set of approaches
            self.approaches = {
                "immediate": base_approaches["immediate"],
                "reasoning": base_approaches["reasoning"],
                "reasoning_2x": base_approaches["reasoning_2x"],
                "reasoning_4x": base_approaches["reasoning_4x"]
            }

    def format_prompt(self, question: str, assistant_content: str = "") -> str:
        """Format prompt using the model's chat template"""
        # Use the tokenizer's chat template for all models
        return self.tokenizer.apply_chat_template([
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ], tokenize=False)

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
        # NOTE: We may need multiple stop tokens
        # llm = Ollama(model="llama3", stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|reserved_special_token"])
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
    
    async def extended_reasoning(self, prompt: str, n_iterations: int, temperature: float = 0, debug: bool = False, use_random: bool = False) -> Dict:
        """Extended reasoning with multiple thinking steps
        
        Args:
            prompt: Input prompt
            n_iterations: Number of additional thinking iterations
            temperature: Sampling temperature
            debug: Whether to print debug info
            use_random: Whether to use random prompts for additional thinking (reasoning_2x_random)
        """
        approach_name = f"reasoning_{n_iterations}x"
        if use_random:
            approach_name = f"reasoning_{n_iterations}x_random"
        
        if debug:
            print("\n" + "="*80)
            print(f"EXTENDED REASONING APPROACH ({n_iterations}x{' random' if use_random else ''})")
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
            
        # Set appropriate stop tokens based on model type
        stop_tokens = ["<|start_header_id|>", "<|eot_id|>"]
        if self.is_huatuo:
            # For HuatuoGPT, we need to stop on the answer token
            stop_tokens = [self.answer_token, "<|eot_id|>"]
            
        # Generate initial reasoning
        output = await self.engine.async_generate(
            prompt=current_prompt,
            sampling_params={
                "stop": stop_tokens,
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
        
        # Random prompts for reasoning_2x_random
        # random_prompts = [
        #     "But wait, let's pause and re-evaluate this. ",  # verification
        #     "But hold on, the question ",                    # correction
        #     "Alright, back to ",                             # backtracking
        #     "Not so fast, "                                  # exploring new paths
        # ]
        random_prompts = [
            "Wait, let me directly compare ",                # comparative reasoning
            # "This would be incorrect if "                  # falsification probe
            "Wait, if ",              # counterfactual exploration
            # "But if I instead consider the dimension of ", # multi-dimensional consideration
            "Wait, a counter-argument is that "                     # dialectical reasoning (Devil's advocate)
            # "If this were true, would this imply that "    # contradiction analysis (reductio ad absurdum)
        ]
        
        # Additional thinking steps
        for i in range(n_iterations):
            if debug:
                print(f"\nStage {i+2}: Extended Thinking ({i+1}/{n_iterations})")
                if use_random:
                    print("Action: Adding think token and random prompt for additional reasoning")
                else:
                    print("Action: Adding think token and Wait for additional reasoning")
                
            # Add think token and prompt for additional reasoning
            if use_random:
                # For reasoning_2x_random, choose a random prompt
                random_prompt = random.choice(random_prompts)
                current_prompt += f"{self.think_token}\n{random_prompt}"
            else:
                # For regular reasoning_2x/4x, use "Wait"
                current_prompt += f"\nWait"
            
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
                    "stop": stop_tokens,  # Allow both stop conditions
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
            
        # Final answer generation
        final_stop_tokens = ["<|eot_id|>"]
        
        output = await self.engine.async_generate(
            prompt=current_prompt,
            sampling_params={
                "stop": final_stop_tokens,
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
    batch_size: int = 32,  # Batch size for parallel processing
    reasoning_approaches: Optional[List[str]] = None  # Optional list of reasoning approaches to use
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
        reasoning_approaches: Optional list of reasoning approaches to use (e.g., ["reasoning", "reasoning_2x_random"])
        
    Returns:
        List of results with scaling approaches for each example
    """
    scaler = TestTimeScaler(engine, tokenizer, reasoning_approaches)
    
    if debug:
        input_data = input_data[:debug_samples]
    
    # Format all inputs upfront
    for item in input_data:
        item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
        item["input_str"] = "Please answer the following multiple-choice question:\n{question}\n{option_str}".format_map(item)
        item["scaling_results"] = []  # Initialize scaling_results upfront
    
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
    
    return final_results