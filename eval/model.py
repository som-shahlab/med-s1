import asyncio
from typing import List, Dict, Any
import json
from tqdm import tqdm

async def process_data_batch(
    engine,
    tokenizer,
    input_data: List[Dict],
    template=None,
    query_prompt=None,
    max_new_tokens=4096,
    temperature=0.0,
    is_use_chat_template=True
) -> List[Dict]:
    """Process a batch of data through the model.
    
    This function handles parallel inference for a batch of items.
    Uses asyncio.gather for parallel processing while being careful about resources.
    
    Args:
        engine: The model engine to use
        tokenizer: The tokenizer to use
        input_data: List of data points to process (a single batch)
        template: Optional template function for formatting prompts
        query_prompt: Optional query prompt template
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        is_use_chat_template: Whether to use chat template
        
    Returns:
        List of results with model outputs
    """
    # Format prompts for this batch
    all_prompts = []
    for item in input_data:
        # Format options string
        options = item['options']
        option_str = "\n".join(f"{k}. {v}" for k, v in options.items())
        
        # Format prompt
        if query_prompt:
            prompt = query_prompt.format(
                question=item['question'],
                option_str=option_str
            )
        else:
            prompt = item['question']
        
        # Apply template if provided
        if template and is_use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = template(messages)
        
        all_prompts.append(prompt)
    
    try:
        # Create tasks for parallel inference
        tasks = []
        for prompt in all_prompts:
            task = asyncio.create_task(
                engine.async_generate(
                    prompt=prompt,
                    sampling_params={
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "top_p": 0.9
                    }
                ),
                name=f"generate_{len(tasks)}"  # Add task names for debugging
            )
            tasks.append(task)
        
        try:
            # Process all prompts in parallel with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout for entire batch
            )
            
            # Store results
            results = []
            for item, prompt, response in zip(input_data, all_prompts, responses):
                if isinstance(response, Exception):
                    # Handle individual task failures
                    print(f"Error processing prompt: {str(response)}")
                    results.append({
                        'output': '',
                        'prompt': prompt,
                        'error': f'Task error: {str(response)}'
                    })
                else:
                    results.append({
                        'output': response['text'],
                        'prompt': prompt,
                        'error': None
                    })
            
            return results
            
        except asyncio.TimeoutError:
            print("Batch processing timed out, cancelling tasks...")
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        
    except Exception as e:
        # Log error but let caller handle it
        print(f"Error in process_data_batch: {str(e)}")
        raise