from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from jinja2 import Template
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import json

@dataclass
class ScalingApproach:
    """Configuration for a test time scaling approach"""
    name: str
    max_tokens: Optional[int]
    ignore_stop: int
    
# Constants for token limits
MAX_NEW_TOKENS = 2048  # Maximum new tokens per LLM call

class TestTimeScaler:
    """Handles test time scaling evaluation with efficient KV cache reuse"""
    
    def __init__(self, llm: LLM, tokenizer: AutoTokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.is_llama = "Llama" in tokenizer.name_or_path
        
        # Set up special tokens
        self.think_token = "<|start_header_id|>think<|end_header_id|>"
        self.answer_token = "<|start_header_id|>answer<|end_header_id|>"
        self.stop_tokens = ["<|eot_id|>"]  # Only stop at actual end, not headers
            
        # Define approaches with max tokens:
        # 1. No reasoning - immediate answer
        # 2. Normal reasoning - up to 2048 tokens
        # 3. 2x thinking - up to 4096 tokens
        # 4. 4x thinking - up to 8192 tokens
        self.approaches = [
            ScalingApproach("immediate", None, 0),  # No reasoning
            ScalingApproach("reasoning", MAX_NEW_TOKENS, 0),  # Normal reasoning
            ScalingApproach("reasoning_2x", MAX_NEW_TOKENS * 2, 2),  # 2x thinking
            ScalingApproach("reasoning_4x", MAX_NEW_TOKENS * 4, 4)   # 4x thinking
        ]

    def count_reasoning_tokens(self, text: str) -> int:
        """Count total number of tokens in all reasoning segments"""
        # Split on think token to get all segments
        think_parts = text.split(self.think_token)
        if len(think_parts) < 2:
            # No think token found
            return 0
            
        total_tokens = 0
        # Process each segment after a think token
        for part in think_parts[1:]:  # Skip first part (before first think)
            # Split on answer token
            answer_parts = part.split(self.answer_token, 1)
            # Get text before answer token (or all text if no answer token)
            reasoning = answer_parts[0].strip()
            
            # Count tokens in this reasoning segment
            segment_tokens = len(self.tokenizer.encode(reasoning))
            total_tokens += segment_tokens
            
            if segment_tokens == 0:
                print(f"\nWARNING: Zero tokens in reasoning segment:")
                print(f"Segment text: {reasoning}")
                
        if total_tokens == 0:
            print(f"\nWARNING: Zero total reasoning tokens:")
            print(f"Full text: {text}")
            
        return total_tokens

    def evaluate_approaches(self, prompt: str, template: Optional[Template] = None, temperature: float = 0, debug: bool = False) -> List[Dict]:
        """Evaluate test time scaling approaches:
        1. Immediate answer - append think+answer tokens immediately
        2. Normal reasoning - let model generate naturally
        3. 2x thinking - ignore end-of-thinking marker twice
        4. 4x thinking - ignore end-of-thinking marker four times
        
        For each approach, we:
        1. Track the number of reasoning tokens (text between think and answer markers)
        2. Use consistent token counting across approaches
        3. Log all prompts and outputs in debug mode
        4. Check for potential issues like conversational loops
        
        The goal is to measure how accuracy changes with reasoning length.
        """
        if template:
            prompt = template.render(
                messages=[{"role": "user", "content": prompt}],
                bos_token=self.tokenizer.bos_token,
                add_generation_prompt=True
            )
            
        results = []
        
        if debug:
            print("\nApproaches:")
            print("-" * 40)
            
        # 1. Immediate answer - force answer token
        if debug:
            print("\n1. Immediate answer")
            print("Initial prompt:")
            print("-" * 30)
            print(prompt)
            print("-" * 30)
            print("Adding think and answer tokens")
            
        immediate_prompt = prompt + self.think_token + "\nI have the answer.\n" + self.answer_token
        if debug:
            print("\nPrompt with answer token:")
            print("-" * 30)
            print(immediate_prompt)
            print("-" * 30)
            print("\nGenerating with stop tokens:", self.tokenizer.encode("<|eot_id|>"))
            
        # Set up sampling parameters for immediate answer
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,
            stop_token_ids=self.tokenizer.encode("<|eot_id|>"),
            skip_special_tokens=False
        )
        output = self.llm.generate(immediate_prompt, sampling_params)[0].outputs[0].text
        full_output = immediate_prompt + output
        
        if debug:
            print("\nModel's output:")
            print("-" * 30)
            print(output)
            print("-" * 30)
            print("\nFull output:")
            print("-" * 30)
            print(full_output)
            print("-" * 30)
            
        # Count reasoning tokens for immediate answer
        reasoning_tokens = self.count_reasoning_tokens(full_output)
        if debug:
            print(f"\nReasoning token count: {reasoning_tokens}")
            
        results.append({
            "approach": "immediate",
            "output": full_output,
            "n_reasoning_tokens": reasoning_tokens
        })
            
        # 2. Normal reasoning - let model generate naturally
        if debug:
            print("\n2. Normal reasoning")
            print("Initial prompt:")
            print("-" * 30)
            print(prompt)
            print("-" * 30)
            print("Adding think token")
            
        base_prompt = prompt + self.think_token
        if debug:
            print("\nPrompt with think token:")
            print("-" * 30)
            print(base_prompt)
            print("-" * 30)
            print("\nGenerating with stop tokens:", self.tokenizer.encode("<|eot_id|>"))
            
        output = self.llm.generate(base_prompt, sampling_params)[0].outputs[0].text
        full_output = base_prompt + output
        
        if debug:
            print("\nModel's output:")
            print("-" * 30)
            print(output)
            print("-" * 30)
            print("\nFull output:")
            print("-" * 30)
            print(full_output)
            print("-" * 30)
            
        # Count only the reasoning tokens (between think and answer)
        reasoning_tokens = self.count_reasoning_tokens(full_output)
        if debug:
            print(f"\nReasoning token count: {reasoning_tokens}")
            
        results.append({
            "approach": "reasoning",
            "output": full_output,
            "n_reasoning_tokens": reasoning_tokens
        })
        
        # 3 & 4. Extended thinking (2x and 4x)
        for approach in self.approaches[2:]:
            if debug:
                print(f"\n{approach.name}")
                print("Adding think token")
                
            # Start with think token
            current_prompt = prompt + self.think_token
            if debug:
                print("Prompt:", current_prompt)
            
            # Set up stop tokens to suppress end token during thinking
            stop_token_ids = self.tokenizer.encode("<|start_header_id|><|end_header_id|>")
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=MAX_NEW_TOKENS,
                min_tokens=1,  # Force at least one token
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False
            )
            
            # Get initial thinking with stop token suppression
            if debug:
                print("\nStarting initial thinking")
                print("Current prompt:")
                print("-" * 30)
                print(current_prompt)
                print("-" * 30)
                print("\nGenerating with stop tokens:", self.tokenizer.encode("<|start_header_id|><|end_header_id|>"))
                
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=MAX_NEW_TOKENS,
                min_tokens=1,
                stop_token_ids=self.tokenizer.encode("<|start_header_id|><|end_header_id|>"),
                skip_special_tokens=False
            )
            
            output = self.llm.generate(current_prompt, sampling_params)[0].outputs[0].text
            current_prompt += output
            
            if debug:
                print("\nModel's initial thinking output:")
                print("-" * 30)
                print(output)
                print("-" * 30)
                print("\nFull prompt after initial thinking:")
                print("-" * 30)
                print(current_prompt)
                print("-" * 30)
                print(f"\nInitial reasoning token count: {len(self.tokenizer.encode(current_prompt))}")
            
            # Track total output tokens from all reasoning calls
            total_reasoning_tokens = len(self.tokenizer.encode(output))  # Initial thinking
            if debug:
                print(f"\nInitial thinking tokens: {total_reasoning_tokens}")
            
            # Wait N times to extend thinking
            for i in range(approach.ignore_stop):
                if debug:
                    print(f"\nWait iteration {i+1}/{approach.ignore_stop}")
                    
                # Add Wait with think header to continue reasoning
                current_prompt += self.think_token + "Wait"
                
                # Get model's continued reasoning
                output = self.llm.generate(
                    current_prompt,
                    SamplingParams(
                        temperature=temperature,
                        top_p=0.9,
                        max_tokens=MAX_NEW_TOKENS,
                        min_tokens=1,
                        stop_token_ids=self.tokenizer.encode("<|start_header_id|><|end_header_id|>"),
                        skip_special_tokens=False
                    )
                )[0].outputs[0].text
                
                # Add output to prompt and count tokens
                current_prompt += output
                output_tokens = len(self.tokenizer.encode(output))
                total_reasoning_tokens += output_tokens
                
                if debug:
                    print(f"  Output tokens this iteration: {output_tokens}")
                    print(f"  Total reasoning tokens so far: {total_reasoning_tokens}")
                if debug:
                    print(f"\nCumulative reasoning tokens after Wait {i+1}: {total_reasoning_tokens}")
                    print(f"New reasoning text from this iteration:")
                    print("-" * 30)
                    print(output)
                    print("-" * 30)
                
                if debug:
                    # Check for repetitive patterns that might indicate a loop
                    if "Yes," in output or "Okay," in output:
                        print("\nWARNING: Detected potential conversational loop")
                    if output.count("Wait") > 1:
                        print("\nWARNING: Detected repeated Wait tokens")
                    
                    # Show token counts for this iteration
                    iteration_tokens = len(self.tokenizer.encode(output))
                    print(f"\nTokens in this iteration: {iteration_tokens}")
                    print(f"Total accumulated reasoning tokens: {total_reasoning_tokens}")
                    
                    # Show full state
                    print("\nFull prompt after output:")
                    print("-" * 30)
                    print(current_prompt)
                    print("-" * 30)
            
            # Get final answer using end token
            if debug:
                print("\nGetting final answer")
                print("Current prompt before answer token:")
                print("-" * 30)
                print(current_prompt)
                print("-" * 30)
                print("Adding answer token")
                
            current_prompt += self.answer_token
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=0.9,
                max_tokens=MAX_NEW_TOKENS,
                min_tokens=0,
                stop_token_ids=self.tokenizer.encode("<|eot_id|>"),
                skip_special_tokens=False
            )
            
            if debug:
                print("\nFinal prompt sent to model:")
                print("-" * 30)
                print(current_prompt)
                print("-" * 30)
                
            final_output = self.llm.generate(current_prompt, sampling_params)[0].outputs[0].text
            full_output = current_prompt + final_output
            
            if debug:
                print("\nModel's final output:")
                print("-" * 30)
                print(final_output)
                print("-" * 30)
                print("\nComplete interaction trace:")
                print("=" * 80)
            
            # Use accumulated reasoning tokens for final count
            final_tokens = total_reasoning_tokens if approach.ignore_stop > 0 else self.count_reasoning_tokens(full_output)
            print(f"\nFinal reasoning token count for {approach.name}: {final_tokens}")
            
            # Validate token counts based on approach
            expected_min = {
                "immediate": 0,
                "reasoning": 100,
                "reasoning_2x": 300,
                "reasoning_4x": 500
            }
            
            if final_tokens < expected_min[approach.name]:
                print(f"WARNING: {approach.name} has unexpectedly low token count: {final_tokens} < {expected_min[approach.name]}")
            
            if approach.name == "reasoning_4x" and len(results) > 2:
                reasoning_2x_tokens = results[-1]["n_reasoning_tokens"]
                if final_tokens <= reasoning_2x_tokens:
                    print(f"WARNING: reasoning_4x tokens ({final_tokens}) not greater than reasoning_2x ({reasoning_2x_tokens})")
            
            # Print running token counts
            print(f"\nToken counts so far:")
            for r in results:
                print(f"  {r['approach']}: {r['n_reasoning_tokens']}")
                
            # Store result with just token count and output
            results.append({
                "approach": approach.name,
                "output": full_output,
                "n_reasoning_tokens": final_tokens
            })
            
            # Print token info
            print(f"\nApproach {approach.name}:")
            print(f"  Total reasoning tokens: {final_tokens}")
            
        return results

def evaluate_test_time_scaling(
    llm: LLM,
    tokenizer: AutoTokenizer,
    input_data: List[Dict],
    template: Optional[Template] = None,
    temperature: float = 0,
    debug: bool = False,
    debug_samples: int = 10,
    batch_size: int = 32  # Added reasonable batch size for vllm
) -> List[Dict]:
    """Run test time scaling evaluation on input data
    
    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer
        input_data: List of evaluation examples
        template: Optional chat template
        temperature: Sampling temperature
        debug: Whether to run in debug mode
        debug_samples: Number of samples to process in debug mode
        
    Returns:
        List of results with scaling approaches for each example
    """
    scaler = TestTimeScaler(llm, tokenizer)
    
    if debug:
        input_data = input_data[:debug_samples]
    
    # Calculate total steps for progress bar
    total_steps = len(input_data) * len(scaler.approaches)
    
    # Setup progress tracking
    final_results = []
    avg_tokens = {approach.name: [] for approach in scaler.approaches}
    
    # Main progress bar showing total progress across all examples and approaches
    with tqdm(total=total_steps, desc="Evaluating", disable=debug) as pbar:
        # Process examples sequentially
        for i, item in enumerate(input_data):
            if debug:
                print(f"\nExample {i+1}/{len(input_data)}")
                
            # Format input
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = "Please answer the following multiple-choice question:\n{question}\n{option_str}".format_map(item)
            
            # Get results for all approaches
            scaling_results = scaler.evaluate_approaches(
                prompt=item["input_str"],
                template=template,
                temperature=temperature,
                debug=debug and i < 10  # Only debug first 10 examples
            )
            
            # Track token counts and update progress
            for result in scaling_results:
                avg_tokens[result['approach']].append(result['n_reasoning_tokens'])
                # Update progress bar for each approach completion
                desc = f"Example {i+1}/{len(input_data)}, " + ", ".join(
                    f"{name}: {sum(tokens)/len(tokens):.0f} tokens"
                    for name, tokens in avg_tokens.items()
                    if tokens
                )
                pbar.set_description(desc)
                pbar.update(1)
            
            item["scaling_results"] = scaling_results
            final_results.append(item)
                
            # Print detailed debug info for first few examples
            if debug and i < debug_samples:
                print(f"\n{'='*80}")
                print(f"Example {i+1}/{debug_samples}")
                print(f"{'='*80}")
                print("\nInput:")
                print("-" * 40)
                print(item['input_str'])
                
                for result in scaling_results:
                    print(f"\n{'-'*40}")
                    print(f"Approach: {result['approach']}")
                    print(f"Reasoning tokens: {result['n_reasoning_tokens']}")
                    print("\nOutput:")
                    print(result['output'])
                
    return final_results