import os
import glob
import pandas as pd
from datasets import Dataset, load_from_disk
from typing import Dict, List, Optional, Tuple, Union
import logging

def get_latest_version(base_dir: Optional[str] = None) -> str:
    """Get the latest curation version directory"""
    if base_dir is None:
        base_dir = os.environ.get('MED_S1K_OUTPUT')
        if not base_dir:
            raise ValueError("MED_S1K_OUTPUT environment variable not set")
    
    versions = glob.glob(os.path.join(base_dir, "*"))
    if not versions:
        raise ValueError(f"No curation versions found in {base_dir}")
    
    # Sort by timestamp in version name (format: version_YYYYMMDD_HHMMSS)
    return max(versions, key=lambda x: x.split('_')[-2:])

def load_curation_outputs(
    version_dir: Optional[str] = None,
    output_type: str = "all"
) -> Union[pd.DataFrame, Dataset, Tuple[pd.DataFrame, Dataset]]:
    """Load curation outputs from a specific version
    
    Args:
        version_dir: Directory containing curation outputs. If None, uses latest version.
        output_type: One of "filtered" (all examples with filtering status),
                    "curated" (selected examples only),
                    "formatted" (HuggingFace dataset ready for training),
                    or "all" (returns both filtered DataFrame and formatted Dataset)
    
    Returns:
        Depending on output_type:
        - "filtered": pd.DataFrame with all examples and their filtering status
        - "curated": pd.DataFrame with only selected examples
        - "formatted": HuggingFace Dataset ready for training
        - "all": Tuple[pd.DataFrame, Dataset] with filtered data and formatted dataset
    """
    if version_dir is None:
        version_dir = get_latest_version()
    
    if output_type in ["filtered", "all"]:
        filtered_path = os.path.join(version_dir, "med_s1k_filtered.parquet")
        filtered_df = pd.read_parquet(filtered_path)
        
        if output_type == "filtered":
            return filtered_df
    
    if output_type == "curated":
        curated_path = os.path.join(version_dir, "med_s1k_curated.parquet")
        return pd.read_parquet(curated_path)
    
    if output_type in ["formatted", "all"]:
        formatted_path = os.path.join(version_dir, "med_s1k_formatted")
        formatted_dataset = load_from_disk(formatted_path)
        
        if output_type == "formatted":
            return formatted_dataset
    
    if output_type == "all":
        return filtered_df, formatted_dataset
    
    raise ValueError(f"Invalid output_type: {output_type}. Must be one of: filtered, curated, formatted, all")

def get_curation_stats(version_dir: Optional[str] = None) -> Dict:
    """Get statistics about the curation process"""
    if version_dir is None:
        version_dir = get_latest_version()
    
    metadata_path = os.path.join(version_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        'version': metadata['version'],
        'timestamp': metadata['timestamp'],
        'original_size': metadata['dataset_info']['original_size'],
        'final_size': metadata['dataset_info']['final_size'],
        'filter_status_counts': metadata['filtering_stats']['filter_status_counts'],
        'specialty_distribution': metadata['filtering_stats']['specialty_distribution'],
        'cot_length_stats': metadata['filtering_stats']['cot_length_stats']
    }

def list_curation_versions(base_dir: Optional[str] = None) -> List[Dict]:
    """List all available curation versions with basic stats"""
    if base_dir is None:
        base_dir = os.environ.get('MED_S1K_OUTPUT')
        if not base_dir:
            raise ValueError("MED_S1K_OUTPUT environment variable not set")
    
    versions = []
    for version_dir in glob.glob(os.path.join(base_dir, "*")):
        try:
            stats = get_curation_stats(version_dir)
            versions.append({
                'path': version_dir,
                'version': stats['version'],
                'timestamp': stats['timestamp'],
                'original_size': stats['original_size'],
                'final_size': stats['final_size']
            })
        except Exception as e:
            logging.warning(f"Failed to load stats for {version_dir}: {e}")
    
    return sorted(versions, key=lambda x: x['timestamp'], reverse=True)