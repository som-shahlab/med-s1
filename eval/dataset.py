import os
import json
from typing import List, Dict, Optional

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
            
        with open(results_json, "r") as f:
            results = json.load(f)
        
        if experiment_name not in results.get("experiments", {}):
            raise ValueError(f"Experiment {experiment_name} not found in results.json")
            
        experiment = results["experiments"][experiment_name]
        
        # Get dataset name from experiment config
        dataset_name = experiment.get("config", {}).get("datasets", {}).get("eval")
        if not dataset_name:
            raise ValueError(f"No evaluation dataset specified in experiment config for {experiment_name}")
        
        # Load config.json to get the dataset path
        med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
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
        else:
            raise ValueError(f"Dataset {dataset_name} has no file_path")
    
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
    
    # Exclude samples if requested
    if exclude_samples:
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