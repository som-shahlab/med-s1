import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime

def setup_logging():
    """Configure logging with consistent format"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Disable HTTP request logging
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)

def load_config() -> Dict:
    """Load configuration from config.json"""
    with open("med-s1/config.json", "r") as f:
        return json.load(f)

def load_experiment_config(experiment_name: str) -> Dict:
    """Load experiment configuration from results.json"""
    with open("med-s1/results.json", "r") as f:
        results = json.load(f)
    if experiment_name not in results["experiments"]:
        raise ValueError(f"Experiment {experiment_name} not found in results.json")
    return results["experiments"][experiment_name]["config"]

def get_output_dir() -> str:
    """Get the output directory from environment"""
    output_dir = os.environ.get('MED_S1K_OUTPUT')
    if not output_dir:
        raise ValueError("MED_S1K_OUTPUT environment variable not set")
    return output_dir

def check_existing_dataset() -> Optional[str]:
    """Check if base dataset exists and return its path"""
    output_dir = get_output_dir()
    filtered_path = os.path.join(output_dir, "med_s1k_filtered.parquet")
    if os.path.exists(filtered_path):
        logging.info(f"Found existing dataset at {filtered_path}")
        return filtered_path
    return None

def get_experiment_dir(experiment_name: str) -> str:
    """Get experiment-specific output directory"""
    output_dir = get_output_dir()
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_intermediate_state(df, stage: str, experiment_name: str):
    """Save intermediate state during processing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        get_experiment_dir(experiment_name),
        f"med_s1k_post_{stage}_{timestamp}.parquet"
    )
    df.to_parquet(output_path)
    logging.info(f"Saved {stage} state to {output_path}")

def update_results_json(experiment_name: str, stage: str, data: Dict):
    """Update results.json with new data for an experiment"""
    with open("med-s1/results.json", "r") as f:
        results = json.load(f)
    
    # Create nested structure if it doesn't exist
    if "results" not in results["experiments"][experiment_name]:
        results["experiments"][experiment_name]["results"] = {}
    
    results["experiments"][experiment_name]["results"][stage] = {
        **data,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("med-s1/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Updated results.json for experiment {experiment_name} {stage}")

def log_stage_results(df, stage: str, filtered_df=None):
    """Log results of a processing stage"""
    logging.info(f"\n=== {stage.title()} Results ===")
    logging.info(f"Total examples: {len(df)}")
    logging.info(f"Kept: {len(df[df['filter_status'] == 'kept'])}")
    
    if filtered_df is not None:
        logging.info(f"Removed: {len(filtered_df)}")
        logging.info("\nRemoval reasons:")
        for reason, count in filtered_df['filter_reason'].value_counts().items():
            logging.info(f"- {reason}: {count}")