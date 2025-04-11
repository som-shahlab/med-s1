import pandas as pd
import os
from tqdm import tqdm

def load_dataset(path: str) -> pd.DataFrame:
    """Load a parquet dataset and print basic info"""
    print(f"\nLoading dataset from {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} examples")
    print(f"Columns: {list(df.columns)}")
    return df

def main():
    # Define paths
    original_path = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/medqa-nejmcr-1k-random-qwen_20250405_183150/med_s1k_curated.parquet"
    transformed_path = "/share/pi/nigam/users/calebwin/hf_cache/med-s1k/medqa-nejmcr-1k-random-nejmcr-qa-reason-qwen-tuned_20250410_132718/med_s1k_curated.parquet"
    
    # Load datasets
    original_df = load_dataset(original_path)
    transformed_df = load_dataset(transformed_path)
    
    # Create comparison dataframe
    comparison_data = []
    print("\nProcessing examples...")
    for idx in tqdm(range(len(original_df))):
        comparison_data.append({
            'Question': original_df.iloc[idx]['Question'],
            'Reasoning': original_df.iloc[idx]['Complex_CoT'],
            'Answer': original_df.iloc[idx]['Response'],
            'Question_Extracted': transformed_df.iloc[idx]['Question'],
            'Reasoning_Extracted': transformed_df.iloc[idx]['Complex_CoT'],
            'Answer_Extracted': transformed_df.iloc[idx]['Response']
        })
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    output_path = "dataset_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Number of examples: {len(comparison_df)}")
    print("\nSample lengths (mean):")
    for col in ['Question', 'Reasoning', 'Answer', 'Question_Extracted', 'Reasoning_Extracted', 'Answer_Extracted']:
        mean_len = comparison_df[col].str.len().mean()
        print(f"{col}: {mean_len:.1f} characters")

if __name__ == "__main__":
    main()