import os
import json

def print_indented(text: str):
    """Prints each line of the string with one tab indentation."""
    for line in text.split('\n'):
        print(f'\t{line}')

def postprocess_output(pred: str) -> str:
    """Postprocess the output of the model."""
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def get_query_prompt(args, experiment=None):
    """Get the appropriate query prompt based on args and experiment config."""
    # Check if experiment config has a prompting approach set to "step"
    step_prompt = False
    if experiment and "config" in experiment:
        step_prompt = experiment["config"].get("prompting", "") == "step"
    
    if args.test_time_scaling or args.strict_prompt:
        base_prompt = "Please answer the following multiple-choice question, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'."
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"
    else:
        base_prompt = "Please answer the following multiple-choice question:"
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"

def resolve_config_reference(config: dict, key: str, results: dict, visited: set = None) -> dict:
    """Recursively resolve 'same as' references in config"""
    if visited is None:
        visited = set()
        
    # Base case: not a reference
    if not isinstance(config.get(key), str) or not config[key].startswith("same as "):
        return config.get(key, {})
        
    # Get referenced experiment
    ref_exp = config[key].replace("same as ", "")
    
    # Check for circular references
    if ref_exp in visited:
        raise ValueError(f"Circular reference detected: {ref_exp}")
    visited.add(ref_exp)
    
    # Get referenced config
    if ref_exp not in results["experiments"]:
        raise ValueError(f"Referenced experiment {ref_exp} not found")
    ref_config = results["experiments"][ref_exp]["config"]
    
    # Recursively resolve if the referenced config also has references
    if isinstance(ref_config.get(key), str) and ref_config[key].startswith("same as "):
        return resolve_config_reference(ref_config, key, results, visited)
        
    return ref_config.get(key, {})

def load_experiment_config(experiment_name: str) -> dict:
    """Load experiment configuration from results.json and resolve references"""
    results_json = os.environ.get('RESULTS_JSON')
    if not results_json:
        raise ValueError("RESULTS_JSON environment variable not set")
        
    with open(results_json, "r") as f:
        results = json.load(f)

    # Handle both old and new format
    experiments = results.get("experiments", results)
    if experiment_name not in experiments:
        raise ValueError(f"Experiment {experiment_name} not found in {results_json}")
    
    # Get raw config
    exp_data = experiments[experiment_name]
    if not isinstance(exp_data, dict) or "config" not in exp_data:
        raise ValueError(f"Invalid experiment data format for {experiment_name}")
        
    config = exp_data["config"]
    if config is None:
        raise ValueError(f"Configuration for experiment {experiment_name} is None")
    
    # Resolve references for each top-level key
    resolved_config = {}
    for key in ["curation", "training_params", "datasets"]:
        resolved_config[key] = resolve_config_reference(config, key, {"experiments": experiments})
    
    # Copy other keys as-is
    for key in config:
        if key not in resolved_config:
            resolved_config[key] = config[key]
    
    return resolved_config