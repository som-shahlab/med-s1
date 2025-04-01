import os
import glob
import pandas as pd
import re
import json

def get_latest_extracts(extract_types):
    """Returns the latest version of each extract type as DataFrames."""
    base_dir = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/"
    results = {}
    
    with open("/share/pi/nigam/users/calebwin/med-s1/results.json", 'r') as f:
        results_json = json.load(f)
    
    for extract_type in extract_types:
        # Find all matching directories for this extract type
        pattern = f"{base_dir}medqa-1k-random-{extract_type}*"
        dirs = glob.glob(pattern)
        
        if dirs:
            # Sort by timestamp (latest first)
            latest_dir = sorted(dirs, key=lambda x: re.search(r'_(\d+_\d+)', x).group(1), reverse=True)[0]
            parquet_path = os.path.join(latest_dir, "med_s1k_curated.parquet")
            
            if os.path.exists(parquet_path):
                results[extract_type] = pd.read_parquet(parquet_path)
                if results[extract_type] is None:
                    print(f"Warning: DataFrame for {extract_type} is None.")
            else:
                print(f"Warning: Parquet file not found for {extract_type} in {latest_dir}")
                # If not found, load from results.json path
                dataset_path = results_json["experiments"][f"medqa-1k-random-{extract_type}"]["results"]["curation"]["dataset_path"]
                parquet_path = os.path.join(os.path.dirname(dataset_path), "med_s1k_curated.parquet")
                results[extract_type] = pd.read_parquet(parquet_path)
    
    return results

def main():
    # Get examples for each syntax type
    extract_types = [
        "list-extract",
        "markdown-extract", 
        "decision-tree-extract",
        "qa-extract",
        "socratic-extract",
        "note-extract",  # SOAP
        "step-extract"
    ]
    
    extracts = get_latest_extracts(extract_types)
    
    # Create markdown content
    markdown_content = "# Examples of Different Reasoning Syntax Types\n\n"
    
    # Add example for each type
    for extract_type in extract_types:
        df = extracts.get(extract_type)
        if df is not None and len(df) > 0:
            # Get first example that has non-empty Complex_CoT
            example = None
            for i in range(len(df)):
                if pd.notna(df.iloc[i]["Complex_CoT"]) and df.iloc[i]["Complex_CoT"].strip():
                    example = df.iloc[i]["Complex_CoT"]
                    break
            
            if example:
                # Format title based on extract type
                title = extract_type.replace("-extract", "").title()
                if title == "Note":
                    title = "SOAP"
                
                markdown_content += f"## {title}\n\n```\n{example}\n```\n\n"
    
    # Write to file
    output_path = "med-s1/data/reasoning_examples.md"
    with open(output_path, "w") as f:
        f.write(markdown_content)
    
    print(f"Examples written to {output_path}")

if __name__ == "__main__":
    main()