# Data Collection

Scripts to collect custom datasets.

# NEJM Case Reports

## Usage

```python
from datasets import Dataset

dataset = Dataset.load_from_disk("./outputs/4_create_dataset")

print(dataset)

> Dataset({
    features: ['id', 'url', 'authors', 'published_date', 'volume', 'issue', 'title', 'question', 'thinking', 'diagnosis_clinical', 'diagnosis_anatomical', 'diagnosis_final', 'diagnosis_pathological', 'diagnosis_psychiatric', 'diagnosis_clinical_and_final', 'diagnosis_diagnosis_and_management', 'diagnosis_diagnosis', 'diagnosis_laboratory'],
    num_rows: 1698
})
```

## How to Generate Dataset

```bash
# Collect URLs.
python 1_collect_urls.py

# Scrape pages.
bash 2_scrape_pages.sh
python 2_scrape_pages.py

# Parse pages.
python 3_parse_pages.py

# Convert pages to HF dataset.
python 4_create_dataset.py
```

### Details on Data Generation

* **Raw Data:** I download every “Case Report” NEJM article (7k total, I only got ~2k) — e.g. https://www.nejm.org/doi/full/10.1056/NEJM199111143252007
* **Reasoning Trace:** I pull all text in the main content (dropping images, tables, etc.) and convert it to Markdown. I save this in the `thinking` column.
* **Labels:** I pull out anything under a subheader with the word Diagnosis/es . I save this as a `diagnosis_` column. I completely drop sections with the header "Dr. XXX's Diagnosis" b/c its just a random doctor opining, and not the final diagnosis.

The resulting HF dataset looks like:

```
Dataset({
    features: ['id', 'url', 'authors', 'published_date', 'volume', 'issue', 'title', 'question', 'thinking', 'diagnosis_clinical', 'diagnosis_anatomical', 'diagnosis_final', 'diagnosis_pathological', 'diagnosis_psychiatric', 'diagnosis_clinical_and_final', 'diagnosis_diagnosis_and_management', 'diagnosis_diagnosis', 'diagnosis_laboratory'],
    num_rows: 1698
})
```