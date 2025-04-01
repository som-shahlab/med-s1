
"""
Usage:

    python 4_create_dataset.py

Purpose:

    - Create a HF dataset of NEJM Case Reports.
"""
import json
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset

def clean_diagnosis_name(name: str) -> str:
    # This regex removes the word "Diagnosis" or "Diagnoses" from the end of the string
    # (?i) makes the pattern case-insensitive, so it matches "Diagnosis", "diagnosis", "DIAGNOSIS", etc.
    # diagnos[ei]s matches either "diagnosis" or "diagnoses"
    # $ ensures it only matches at the end of the string
    cleaned: str = re.sub(r' (?i)diagnos[ei]s$', '', name).strip()
    # Do some custom filtering
    if cleaned.lower() == "anatomic":
        cleaned = "Anatomical"
    elif cleaned.lower() == "clincal":
        cleaned = "Clinical"
    elif cleaned.lower() == "integrated anatomical":
        cleaned = "Anatomical"
    elif cleaned.lower() == "final pathological":
        cleaned = "Pathological"
    elif cleaned.lower() == "anantomical":
        cleaned = "Anatomical"

    return cleaned.lower().replace(" ", "_")

def is_doctor_diagnosis(name: str) -> bool:
    """
    There are lots of diagnoses within NEJM case reports that are just a random doctor opining, 
    e.g. "Dr. Taylor's Diagnosis" or "Dr. Smith's Diagnosis". Let's remove these and just keep the
    "Clinical" or "Final" etc. diagnoses.
    """
    return 'dr.' in name.lower() or "'" in name.lower() or 'drs.' in name.lower() or "’" in name.lower()

# Example usage:
if __name__ == "__main__":
    n_workers: int = 10
    
    path_to_input_dir: str = "outputs/3_parse_pages/"
    path_to_output_dir: str = "outputs/4_create_dataset/"
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load parsed JSONs
    paths_to_jsons: List[str] = sorted([ os.path.join(path_to_input_dir, f) for f in os.listdir(path_to_input_dir) if f.endswith(".json") ])
    print(f"Found {len(paths_to_jsons)} JSONs in {path_to_input_dir}.")
    
    # Print out all types of diagnoses
    diagnoses = set()
    for path_to_json in paths_to_jsons:
        json_data = json.load(open(path_to_json, "r"))
        if json_data['bodymatter'] and json_data['bodymatter'].get('diagnosis_sections', {}):
            diagnoses.update([ 
                clean_diagnosis_name(x)
                for x in json_data['bodymatter'].get('diagnosis_sections', {}).keys()
            ])
    non_doctor_diagnoses = [ d for d in diagnoses if not is_doctor_diagnosis(d) ]
    print(f"# of unique diagnoses: {len(diagnoses)}")
    print(f"# of non-doctor diagnoses: {len(non_doctor_diagnoses)}")
    print("Non-doctor diagnoses:", non_doctor_diagnoses)

    dataset = []
    unique_diagnosis_keys = set()
    for path_to_json in tqdm(paths_to_jsons):
        json_data = json.load(open(path_to_json, "r"))
        if json_data['bodymatter'] is None:
            continue
        # Ignore if no presentation of case (b/c no question)
        if json_data['bodymatter'].get('intro_sections') is None or json_data['bodymatter'].get('intro_sections').get('presentation_of_case') is None:
            continue
        # Ignore if no diagnosis sections (b/c no labels)
        if json_data['bodymatter'].get('diagnosis_sections') is None:
            continue
        
        # Clean up presentation of case (this will serve as our question)
        presentation_of_case: str = json_data['bodymatter'].get('intro_sections', {}).get('presentation_of_case', '')

        # Clean up diagnoses (these will serve as our labels)
        diagnosis_sections: Dict[str, str] = {
            f"diagnosis_{clean_diagnosis_name(k)}" : v
            for k, v in json_data.get('bodymatter', {}).get('diagnosis_sections', {}).items()
            if not is_doctor_diagnosis(k) # Ignore doctor diagnoses
        }
        unique_diagnosis_keys.update(diagnosis_sections.keys())

        # Create dataset
        dataset.append({
            'id' : json_data['id'],
            'url' : json_data['url'],
            # metadata
            'authors' : json_data['frontmatter'].get('authors', []),
            'published_date' : json_data['frontmatter'].get('published_date', ''),
            'volume' : json_data['frontmatter'].get('volume', ''),
            'issue' : json_data['frontmatter'].get('issue', ''),
            'title' : json_data['frontmatter'].get('title', ''),
            # bodymatter
            'question' : presentation_of_case,
            'thinking' : json_data['bodymatter'].get('markdown', ''),
            **diagnosis_sections,
        })

    # Save as local HF dataset
    # force first entry to have all diagnosis_ keys, otherwise `from_list` will drop them
    dataset[0] = { **dataset[0], **{ k: None for k in unique_diagnosis_keys } }
    hf_dataset = Dataset.from_list(dataset)
    print(hf_dataset)
    hf_dataset.save_to_disk(path_to_output_dir)

