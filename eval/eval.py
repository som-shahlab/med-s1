import argparse
import openai
import os
import json
from tqdm import tqdm
from jinja2 import Template
from transformers import AutoTokenizer
from scorer import get_results
from typing import List, Dict, Tuple

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--path_to_eval_json', type=str, required=True, help='Path to the evaluation data (i.e. `data/eval_data.json`)')
    parser.add_argument('--path_to_output_dir', type=str, default='./results', help='Path to the output directory.')
    parser.add_argument('--max_new_tokens', type=int, default=2000, help='Maximum number of new tokens to generate.')
    parser.add_argument('--max_tokens', type=int, default=-1, help='Maximum number of tokens to generate. If -1, no truncation is performed.')
    parser.add_argument('--use_chat_template',type=bool, default=True, help='Use chat template.')
    parser.add_argument('--strict_prompt', action="store_true", help='Use strict prompt.')
    parser.add_argument('--task', type=str, default='api', help='Task name.')
    parser.add_argument('--port', type=int, default=30000, help='Port number.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')    
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature.')
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

def call_model(client: openai.Client, prompts: List[str], model: str, tokenizer: AutoTokenizer, template: Template, max_new_tokens: int = 50, temperature: float = 0, is_print_example: bool = False, is_use_chat_template: bool = False, max_tokens: int = -1) -> Tuple[List[str], List[str]]:
    """Call the model to get the predicted output.
    Args:
        prompts (List[str]): The prompts to call the model with.
        model (str): The model to call.
        max_new_tokens (int): The maximum number of new tokens to generate.
        is_print_example (bool): Whether to print an example.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the postprocessed predicted outputs and the raw predicted outputs.
    """
    if is_print_example:
        print("Raw prompt:")
        print("```")
        print_indented(prompts[0])
        print("```")

    preds: List[str] = []
    if is_use_chat_template:
        prompts = [template.render(messages=[{"role": "user", "content": p}],
                                   bos_token=tokenizer.bos_token, # '<|begin_of_text|>'
                                   add_generation_prompt=True) 
                   for p in prompts]
    
    if max_tokens > 0:
        new_prompts: List[str] = []
        for prompt in prompts:
            input_ids: List[int] = tokenizer.encode(prompt,add_special_tokens= False)
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

    response = client.completions.create(
        model="default",
        prompt=prompts,
        temperature=temperature, 
        top_p=0.9, 
        max_tokens=max_new_tokens
    )
    raw_preds: List[str] = [x.text for x in response.choices]
    preds: List[str] = [postprocess_output(pred) for pred in raw_preds]
    
    if is_print_example:
        print("Postprocessed predicted output:")
        print("```")
        print_indented(preds[0])
        print("```")
        
    return preds, raw_preds

def main():
    args = parse_args()
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    print(f"Using local API server at port {args.port}")
    client = openai.Client(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    if args.use_chat_template:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template: Template = Template(tokenizer.chat_template)

    input_data: List[Dict] = load_file(args.path_to_eval_json)
    model: str = None

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

        preds, _ = call_model(client, processed_batch, model=model, tokenizer=tokenizer, template=template, max_new_tokens=args.max_new_tokens, is_print_example=is_print_example, temperature=args.temperature, is_use_chat_template=args.use_chat_template, max_tokens=args.max_tokens)

        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)

    # Save outputs
    task_name: str = os.path.split(args.model_name)[-1]
    task_name = task_name + os.path.basename(args.path_to_eval_json).replace('.json','') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
    file_name: str = f'{task_name}.json'
    path_to_output: str = os.path.join(args.path_to_output_dir, file_name)
    with open(path_to_output,'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)

    # Score outputs
    get_results(path_to_output)


if __name__ == "__main__":
    main()
