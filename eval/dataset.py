import os
import json
from typing import List, Dict, Optional

def load_eval_dataset(experiment_name: str, path_to_eval_json: Optional[str] = None) -> List[Dict]:
    """Load evaluation dataset from file or experiment config.
    
    Args:
        experiment_name: Name of experiment from results.json
        path_to_eval_json: Optional path to evaluation data file
        
    Returns:
        List of evaluation samples
    """
    # If path provided, load directly
    if path_to_eval_json:
        if not os.path.exists(path_to_eval_json):
            raise ValueError(f"Evaluation data file not found: {path_to_eval_json}")
        
        with open(path_to_eval_json, 'r') as f:
            data = json.load(f)
            
        # Handle both formats:
        # 1. List of samples
        # 2. Dict with source keys
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Flatten dict into list, adding source to each sample
            samples = []
            for source, source_data in data.items():
                for sample in source_data:
                    # Always set source to match the source key
                    sample['source'] = source
                    samples.append(sample)
            return samples
        else:
            raise ValueError(f"Invalid evaluation data format in {path_to_eval_json}")
    
    # Otherwise, try to get dataset from experiment config
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
    
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    if experiment_name not in results['experiments']:
        raise ValueError(f"Experiment {experiment_name} not found in {results_json}")
    
    experiment = results['experiments'][experiment_name]
    # Get experiment config
    config_data = experiment.get('config', {})
    datasets = config_data.get('datasets')
    
    # Handle special case where datasets is "same as base"
    if isinstance(datasets, str) and datasets == "same as base":
        # Get datasets from base experiment
        base_experiment = results['experiments']['base']
        eval_dataset = base_experiment['config']['datasets']['eval']
    elif isinstance(datasets, dict) and datasets.get('eval'):
        eval_dataset = datasets['eval']
    else:
        raise ValueError(f"No evaluation dataset specified in config for {experiment_name}")
    
    # Load dataset based on config
    med_s1_dir = os.environ.get('MED_S1_DIR')
    if not med_s1_dir:
        raise ValueError("MED_S1_DIR environment variable not set")
    
    with open(os.path.join(med_s1_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    if eval_dataset not in config.get('eval_datasets', {}):
        raise ValueError(f"Dataset {eval_dataset} not found in config.json")
    
    dataset_config = config['eval_datasets'][eval_dataset]
    
    # Load from file path
    if 'file_path' in dataset_config:
        file_path = dataset_config['file_path'].replace('${MED_S1_DIR}', med_s1_dir)
        if not os.path.exists(file_path):
            raise ValueError(f"Dataset file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle both formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            samples = []
            for source, source_data in data.items():
                for sample in source_data:
                    # Always set source to match the source key
                    sample['source'] = source
                    samples.append(sample)
            return samples
        else:
            raise ValueError(f"Invalid dataset format in {file_path}")
    else:
        raise ValueError(f"Unsupported dataset config for {eval_dataset}")