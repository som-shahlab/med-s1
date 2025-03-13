#!/usr/bin/env python3
"""
Script to update results.json with training results.
This is extracted from sft_carina.sh to make it easier to maintain.
"""

import os
import sys
import json
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Update results.json with training results")
    parser.add_argument("--experiment_name", required=True, help="Name of experiment from results.json")
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--results_json", required=True, help="Path to results.json")
    args = parser.parse_args()

    # Import path_utils for updating results.json
    med_s1_dir = os.environ.get('MED_S1_DIR', '/share/pi/nigam/users/calebwin/med-s1')
    sys.path.append(os.path.join(med_s1_dir, 'data'))
    
    try:
        from utils.path_utils import update_results_json
    except ImportError:
        print("Error: Could not import path_utils. Make sure MED_S1_DIR is set correctly.")
        sys.exit(1)
    
    # Create paths dict for update_results_json
    paths = {
        'checkpoint': os.path.abspath(args.model_path)
    }
    
    # Update results.json
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    try:
        update_results_json(
            results_json_path=args.results_json,
            experiment_name=args.experiment_name,
            stage="training",
            paths=paths,
            timestamp=timestamp
        )
        print(f"Successfully updated results.json for experiment {args.experiment_name}")
    except Exception as e:
        print(f"Error updating results.json: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()