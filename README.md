# med-s1: Medical Reasoning Experiments

This project explores various methods for training and evaluating medical reasoning models.

## Overview

The core workflow consists of three stages:

1.  **Curation:** Selecting and formatting data for training.
2.  **Training:** Fine-tuning a language model on the curated data.
3.  **Evaluation:** Assessing the performance of the trained model.

Experiments are defined and executed through `results.json`. This file specifies the configuration for each experiment, including the model, datasets, training parameters, and curation methods.

## Running Experiments

### Batch Scripts

The following batch scripts are provided for running experiments:

*   `curate_all.sh`: Runs the curation stage for all experiments defined in `results.json`.
*   `train_all.sh`: Runs the training stage for all experiments defined in `results.json`.
*   `eval_all.sh`: Runs the evaluation stage for all experiments defined in `results.json`.

### Individual Commands

You can also execute each stage for a single experiment using the following commands:

*   **Curation:** `bash curate_med_s1k.sh <experiment_name>`
*   **Training:** `sbatch train/sft_carina.sh <experiment_name>`
*   **Evaluation:** `sbatch eval/eval.sh <experiment_name>`

Replace `<experiment_name>` with the name of the experiment defined in `results.json`.

### Authentication

For curation, you may need to authenticate with Google Cloud using `gcloud auth application-default login` in order to use the Gemini models.

### Configuration

The `config.sh` file may need to be edited to include your WANDB\_API\_KEY and HUGGING\_FACE\_HUB\_TOKEN.

### Plotting Results

The `plot_model_comparison.py` script is used to generate accuracy plots for different methods. This is the final step in the evaluation process.

## Implementing a New Dataset

To implement a new dataset, you need to:

1.  **Add a new entry to the `config.json` file**:
    *   In the `"train_datasets"` section, define the dataset's properties, such as its Hugging Face path (`hf_path`), configuration (`hf_config`), and split (`hf_split`).
    *   Example for a MedQA dataset:
        ```json
        "medqa": {
            "hf_path": "path/to/medqa",
            "hf_config": "subset",
            "hf_split": "train"
        }
        ```
    *   Alternatively, you can specify a local file path (`file_path`) to a JSON, Parquet, or directory containing a saved Hugging Face Dataset.
2.  **Modify the curation pipeline (if necessary)**:
    *   The `curate_med_s1k_new.py` script loads the dataset using the `load_base_dataset` function. This function handles loading datasets from Hugging Face or local files.
    *   If the dataset requires specific preprocessing or formatting, you may need to modify the `load_base_dataset` function or create a new curation method in the `curation_methods` directory.
    *   For example, the MedQA dataset is loaded using the `load_base_dataset` function, which reads the dataset from a Hugging Face path specified in `config.json`. The function then initializes metadata columns for filtering and selection.
3.  **Reference the new dataset in `results.json`**:
    *   In the experiment configuration, specify the name of the new dataset in the `"datasets"` section under the `"curate"` key.

## Adding a New Reasoning Trace Style Transformation

To add a new reasoning trace style or perturbation, you need to:

1.  Implement the new formatting logic in `clinical_formatting.py`. This file contains functions for transforming chain-of-thought reasoning into various formats.
2.  Modify the curation pipeline to use the new formatting method.
3.  If modifying the prompt to improve reasoning trace quality, you can use the `restore` flag in the `results.json` file.

## Adding an Experiment that Finetunes a Different Model

To add an experiment that finetunes a different model, you need to:

1.  **Add a new entry to the `config.json` file**:
    *   In the `"models"` section, define the model's properties, such as its Hugging Face path (`hf_path`) and `max_length`.
    *   Example:
        ```json
        "my_new_model": {
            "hf_path": "huggingface/path/to/my_new_model",
            "max_length": 2048
        }
        ```
2.  **Update `results.json`**:
    *   In the experiment configuration, specify the name of the new model in the `"config"` section under the `"model_key"` key.
    *   Example:
        ```json
        "my_new_experiment": {
            "description": "Experiment with my new model",
            "config": {
                "model_key": "my_new_model",
                "datasets": "same as base-step-prompt"
            }
        }
        ```
3.  **Handle different chat formats (in `data/utils/formatting.py`)**:
    *   The `format_for_training` function in `data/utils/formatting.py` is responsible for formatting the data for training.
    *   If the new model has a different chat format, you will need to adjust the prompt formatting in the `format_for_training` function to match the expected input format of the model. This might involve changing the prompt structure, adding special tokens, or modifying the way the input and output are concatenated.
    *   For example, you might add a conditional statement that checks the `model_name` and applies the appropriate formatting logic based on the model's chat format.
4.  **Ensure compatibility with `train/sft.py`**:
    *   The `train/sft.py` script loads the model and tokenizer using `AutoModelForCausalLM` and `AutoTokenizer`. Ensure that the new model is compatible with these classes.
    *   If the new model requires specific training parameters or configurations, you may need to modify the `train/sft.py` script accordingly.

## Michael's Notes

```bash
conda activate /local-scratch/nigam/users/mwornow/envs/meds1

bash curate_med_s1k.sh <experiment_name>
bash train/sft_carina.sh <experiment_name>
bash eval/eval.sh <experiment_name>
```