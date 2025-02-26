import argparse
from vllm import LLM, SamplingParams
import os
import json
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from scorer import get_results
from typing import List, Dict, Tuple
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of experiment from results.json')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--path_to_eval_json', type=str, required=True, help='Path to the evaluation data')
    parser.add_argument('--path_to_output_dir', type=str, default='./results', help='Path to the output directory')
    parser.add_argument('--max_new_tokens', type=int, default=2000, help='Maximum number of new tokens to generate')
    parser.add_argument('--max_tokens', type=int, default=-1, help='Maximum number of tokens to generate. If -1, no truncation is performed')
    parser.add_argument('--use_chat_template', type=bool, default=True, help='Use chat template')
    parser.add_argument('--strict_prompt', action="store_true", help='Use strict prompt')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    return parser.parse_args()

def print_indented(text: str):
    """Prints each line of the string with one tab indentation."""
    for line in text.split('\n'):
        print(f'\t{line}')

def postprocess_output(pred: str) -> str:
    """Postprocess the output of the model.
    Args:
        pred (str): The predicted output of the model.
    Returns:
        str: The postprocessed predicted output.
    """
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp: str) -> List[Dict]:
    """Load the evaluation data from a JSON file.
    Args:
        input_fp (str): The path to the JSON file containing the evaluation data.

    Returns:
        list: A list of dictionaries containing the evaluation data.

    Each example in the returned list looks like this:
        {
            'question': 'Which of the following is not true for myelinated nerve fibers:', 
            'options': {
                'A': 'Impulse through myelinated fibers is slower than non-myelinated fibers', 
                'B': 'Membrane currents are generated at nodes of Ranvier', 
                'C': 'Saltatory conduction of impulses is seen', 
                'D': 'Local anesthesia is effective only when the nerve is not covered by myelin sheath'
            }, 
            'answer_idx': 'A', 
            'answer': 'Impulse through myelinated fibers is slower than non-myelinated fibers', 
            'source': 'MedMCQA_validation'
        }
        
    Count of each example in the HuatuoGPT-O1 evaluation dataset:
        Counter({
            'MedMCQA_validation': 4183, 
            'MMLU-Pro_Medical_test': 1535, 
            'MedQA_USLME_test': 1273, 
            'PubMedQA_test': 1000, 
            'GPQA_Medical_test': 390
        })
    """
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data

def call_model(llm: LLM, prompts: List[str], tokenizer: AutoTokenizer, template: Template, max_new_tokens: int = 50, temperature: float = 0, is_print_example: bool = False, is_use_chat_template: bool = False, max_tokens: int = -1) -> Tuple[List[str], List[str]]:
    """Call the model to get the predicted output using vllm.
    Args:
        llm (LLM): The vllm LLM instance
        prompts (List[str]): The prompts to call the model with
        tokenizer (AutoTokenizer): The tokenizer to use
        template (Template): The chat template to use
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Sampling temperature
        is_print_example (bool): Whether to print an example
        is_use_chat_template (bool): Whether to use the chat template
        max_tokens (int): Maximum total tokens (-1 for no limit)

    Returns:
        Tuple[List[str], List[str]]: Tuple of (processed predictions, raw predictions)
    """
    if is_print_example:
        print("Raw prompt:")
        print("```")
        print_indented(prompts[0])
        print("```")

    if is_use_chat_template:
        prompts = [template.render(messages=[{"role": "user", "content": p}],
                                   bos_token=tokenizer.bos_token,
                                   add_generation_prompt=True)
                   for p in prompts]
    
    if max_tokens > 0:
        new_prompts: List[str] = []
        for prompt in prompts:
            input_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=False)
            if len(input_ids) > max_tokens:
                input_ids = input_ids[:max_tokens]
                new_prompts.append(tokenizer.decode(input_ids))
            else:
                new_prompts.append(prompt[-max_tokens:])
        prompts = new_prompts

    if is_print_example:
        print("Tokenized prompt:")
        print("```")
        print_indented(prompts[0])
        print("```")

    # Set up vllm sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Generate completions using vllm
    outputs = llm.generate(prompts, sampling_params)
    raw_preds = [output.outputs[0].text for output in outputs]
    preds = [postprocess_output(pred) for pred in raw_preds]
    
    if is_print_example:
        print("Postprocessed predicted output:")
        print("```")
        print_indented(preds[0])
        print("```")
        
    return preds, raw_preds

def main():
    args = parse_args()
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Initialize vllm model
    print(f"Initializing vllm with model: {args.model_path}")
    print(f"Using tensor parallel size: {args.tensor_parallel_size}")
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    template = Template(tokenizer.chat_template) if args.use_chat_template else None

    input_data: List[Dict] = load_file(args.path_to_eval_json)

    final_results: List[Dict] = []
    if args.strict_prompt:
        query_prompt = "Please answer the following multiple-choice questionss, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'.\n{question}\n{option_str}"
    else:
        query_prompt = "Please answer the following multiple-choice question:\n{question}\n{option_str}"        

    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch: List[Dict] = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if len(batch) == 0:
            break

        # Format inputs to LLM
        for item in batch:
            item['option_str'] = '\n'.join([ f'{op}. {ans}' for op,ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)
        processed_batch: List[str] = [ item["input_str"] for item in batch]

        # Always print the first example for sanity checking
        is_print_example: bool = idx == 0

        preds, _ = call_model(
            llm=llm,
            prompts=processed_batch,
            tokenizer=tokenizer,
            template=template,
            max_new_tokens=args.max_new_tokens,
            is_print_example=is_print_example,
            temperature=args.temperature,
            is_use_chat_template=args.use_chat_template,
            max_tokens=args.max_tokens
        )

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    # Save outputs
    model_name: str = os.path.split(args.model_path)[-1]
    task_name: str = model_name + os.path.basename(args.path_to_eval_json).replace('.json','') + ('_strict-prompt' if args.strict_prompt else '')
    file_name: str = f'{task_name}.json'
    path_to_output: str = os.path.join(args.path_to_output_dir, file_name)
    with open(path_to_output,'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # Score outputs and get metrics
    metrics = get_results(path_to_output)
    
    # Update results.json with eval results and metrics
    with open("med-s1/results.json", "r") as f:
        results = json.load(f)
    
    results["experiments"][args.experiment_name]["results"]["eval"] = {
        "results_path": os.path.abspath(path_to_output),
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    with open("med-s1/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
