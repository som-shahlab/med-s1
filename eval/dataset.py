import os
import json
from typing import List, Dict, Optional
from resolve_eval_config import resolve_eval_config
from datasets import load_dataset, Dataset
import random


def process_gpqa_diamond_as_eval(dataset: Dataset) -> List[Dict]:
    """Process the Gpqa Diamond dataset as an evaluation dataset."""
    assert 'Validator Revision Suggestion_EV_1' in dataset.features, f"A column named 'Validator Revision Suggestion_EV_1' should be present in the dataset. Are you sure you loaded the GPQA dataset?"
    input_data: List[Dict] = []
    for da in dataset:
        # Following procedure here: https://github.com/simplescaling/s1/blob/bbb6b50138cbd07dd818cce0060eedafebdbd8fd/eval/lm-evaluation-harness/lm_eval/tasks/gpqa/openai/utils.py#L221
        choices = [
            da['Incorrect Answer 1'],
            da['Incorrect Answer 2'],
            da['Incorrect Answer 3'],
            da['Correct Answer']
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(da["Correct Answer"])
        options = {
            "A" : choices[0],
            "B" : choices[1],
            "C" : choices[2],
            "D" : choices[3],
        }
        input_data.append({
            'question' : da['Question'],
            'options' : options,
            'answer_idx' : f"{chr(65 + correct_answer_index)}",
            'answer' : da['Correct Answer'],
            'source' : 'gpqa-diamond'
        })
    return input_data

def load_eval_dataset(experiment_name: str, path_to_eval_json: Optional[str] = None, exclude_samples: bool = True) -> List[Dict]:
    """Load the evaluation dataset either from a specified path or from config.
    
    Args:
        experiment_name: Name of experiment from results.json
        path_to_eval_json: Optional path to evaluation data file. If not provided, will use dataset from config
        exclude_samples: Whether to exclude samples from eval_data_samples.json
        
    Returns:
        List of dictionaries containing evaluation data
    """
    # Get dataset path from config if not provided
    if not path_to_eval_json:
        results_json = os.environ.get('RESULTS_JSON')
        if not results_json:
            raise ValueError("RESULTS_JSON environment variable not set")
        # Get experiment config with resolved references
        resolved = resolve_eval_config(experiment_name, results_json)
        experiment = resolved['config']
        
        # Get dataset name from experiment config
        dataset_name = experiment.get('datasets', {}).get('eval')
        if not dataset_name:
            raise ValueError(f"No evaluation dataset specified in experiment config for {experiment_name}")
        
        # Load config.json to get the dataset path
        assert os.environ.get('MED_S1_DIR') is not None, "MED_S1_DIR environment variable not set"
        med_s1_dir = os.environ.get('MED_S1_DIR')
        with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        if dataset_name not in config.get("eval_datasets", {}):
            raise ValueError(f"Dataset {dataset_name} not found in config.json")
        
        dataset_config = config["eval_datasets"][dataset_name]
        
        # Get file path from dataset config
        if "file_path" in dataset_config:
            path_to_eval_json = dataset_config["file_path"]
            # Replace environment variables
            path_to_eval_json = path_to_eval_json.replace("${MED_S1_DIR}", med_s1_dir)
            # Load and process the data
            with open(path_to_eval_json, 'r') as f:
                data = json.load(f)
            input_data = []
            if isinstance(data, list):
                data = {'normal': data}
            for k,v in data.items():
                for da in v:
                    da['source'] = k
                input_data.extend(v)
        elif "hf_path" in dataset_config:
            # Load and process the data
            dataset = load_dataset(dataset_config["hf_path"], dataset_config['hf_config'], split=dataset_config["hf_split"])
            if dataset_name == 'gpqa-diamond':
                input_data = process_gpqa_diamond_as_eval(dataset)
            else:
                raise NotImplementedError(f"Dataset `{dataset_name}` has an `hf_path` attribute, but no processing function is implemented yet. Please implement a function to process the dataset and add it to the `eval.dataset.py` file.")
        else:
            raise ValueError(f"Dataset {dataset_name} has no `file_path` or `hf_path` attribute in config.json")

    # Exclude samples if requested
    if path_to_eval_json and exclude_samples:
        # Get the directory of the input file
        input_dir = os.path.dirname(path_to_eval_json)
        samples_path = os.path.join(input_dir, 'eval_data_samples.json')
        try:
            with open(samples_path, 'r') as f:
                samples = json.load(f)
            
            # Create a set of question strings for faster lookup
            sample_questions = {sample.get('question', '') for sample in samples}
            
            # Filter out samples
            original_count = len(input_data)
            input_data = [item for item in input_data if item.get('question', '') not in sample_questions]
            excluded_count = original_count - len(input_data)
            print(f"Excluded {excluded_count} samples from evaluation data")
        except FileNotFoundError:
            print(f"Warning: Samples file {samples_path} not found. No samples will be excluded.")
    
    return input_data