import asyncio
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import Counter
from scorer import match_choice

async def process_data_batch(
    engine,
    tokenizer,
    input_data: List[Dict],
    template,
    query_prompt: str,
    max_new_tokens: int,
    temperature: float,
    is_use_chat_template: bool,
    max_tokens: int,
    batch_size: int = 256
) -> List[Dict]:
    """Process a batch of data through the model."""
    results = []
    total_batches = (len(input_data) + batch_size - 1) // batch_size
    
    print(f"Processing {len(input_data)} examples in {total_batches} batches (size {batch_size})...")
    
    for i in tqdm(range(0, len(input_data), batch_size), desc="Processing batches"):
        batch = input_data[i:i + batch_size]
        current_batch = i // batch_size + 1
        print(f"\nBatch {current_batch}/{total_batches} ({len(batch)} examples)")
        
        # Format batch
        for item in batch:
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)
        
        # Process batch
        prompts = [item["input_str"] for item in batch]
        if is_use_chat_template:
            prompts = [template.render(messages=[{"role": "user", "content": p}],
                                    bos_token=tokenizer.bos_token,
                                    add_generation_prompt=True)
                      for p in prompts]
        
        # Generate completions using sglang
        tasks = [
            engine.async_generate(
                prompt=prompt,
                sampling_params={
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "top_p": 0.9
                }
            )
            for prompt in prompts
        ]
        outputs = await asyncio.gather(*tasks)
        preds = [output['text'] for output in outputs]
        
        # Store results
        for item, pred in zip(batch, preds):
            if len(pred) > 0:
                item_copy = item.copy()
                item_copy["output"] = pred
                results.append(item_copy)
    
    return results

async def process_gpqa_with_multiple_runs(
    engine,
    tokenizer,
    gpqa_data: List[Dict],
    template,
    query_prompt: str,
    max_new_tokens: int,
    temperature: float,
    is_use_chat_template: bool,
    max_tokens: int,
    batch_size: int,
    num_runs: int
) -> List[Dict]:
    """Process GPQA data with multiple runs and majority voting."""
    print(f"\nProcessing {len(gpqa_data)} GPQA examples with {num_runs} runs each...")
    gpqa_results = []
    
    # Run GPQA data multiple times
    for run in range(num_runs):
        print(f"\nGPQA Run {run+1}/{num_runs}")
        run_results = await process_data_batch(
            engine=engine,
            tokenizer=tokenizer,
            input_data=gpqa_data,
            template=template,
            query_prompt=query_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            is_use_chat_template=is_use_chat_template,
            max_tokens=max_tokens,
            batch_size=batch_size
        )
        
        # Store results for this run
        for i, result in enumerate(run_results):
            if run == 0:
                # First run, initialize the result with runs array
                result["runs"] = [{"output": result["output"]}]
                gpqa_results.append(result)
            else:
                # Add this run's output to the existing result
                gpqa_results[i]["runs"].append({"output": result["output"]})
    
    # Process the multiple runs to get averaged results
    for result in gpqa_results:
        # Get all outputs from runs
        outputs = [run["output"] for run in result["runs"]]
        
        # Get all parsed answers
        answers = []
        for output in outputs:
            ans, ans_type = match_choice(output, result["options"])
            answers.append(ans[-1])  # Use the last answer (most likely the final answer)
        
        # Count occurrences of each answer
        answer_counts = Counter(answers)
        
        # Get the most common answer
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Use the output from the first run where the answer matches the most common answer
        for i, run in enumerate(result["runs"]):
            ans, _ = match_choice(run["output"], result["options"])
            if ans[-1] == most_common_answer:
                result["output"] = run["output"]
                result["majority_vote"] = most_common_answer
                result["vote_counts"] = dict(answer_counts)
                result["confidence"] = answer_counts[most_common_answer] / num_runs
                break
    
    return gpqa_results