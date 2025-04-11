# Evaluation Data Sources

This directory contains scripts and data for generating evaluation datasets. The evaluation data consists of multiple sources:

## Base Sources

1. **MedMCQA_validation**
   - Medical multiple choice questions from MedMCQA validation set
   - Tests general medical knowledge and clinical reasoning

2. **MedQA_USLME_test**
   - Questions from USMLE Step exams
   - Tests medical knowledge at physician licensing level

3. **PubMedQA_test**
   - Questions derived from PubMed abstracts
   - Tests ability to reason about medical research

4. **MMLU-Pro_Medical_test**
   - Medical subset of MMLU-Pro test set
   - Tests advanced medical knowledge

5. **GPQA_Medical_test**
   - Medical subset of GPQA test set
   - Tests general practitioner level knowledge

## Generated Sources

6. **MedDS**
   - Subset of base sources filtered for care planning and diagnosis questions
   - Questions must involve specific patient cases
   - Tests clinical decision making and diagnostic reasoning

7. **MedDS_NOTA**
   - "None of the Above" versions of MedDS questions
   - Original correct answer replaced with "None of the above" as option D
   - Other options preserved in order as A-C
   - Tests robustness of clinical reasoning

8. **NEJMCRMC**
   - Multiple choice questions generated from NEJM case reports
   - Uses cases with >8192 tokens (longer, more complex cases)
   - Generated using:
     * nejmcr_qa to create question/answer
     * nejmcr_reason and nejmcr_clean for reasoning
     * generate_mc_options for challenging alternatives
   - Tests complex clinical reasoning on real medical cases

## Generation Scripts

- `generate_medds_eval.py`: Creates MedDS and MedDS_NOTA datasets
- `generate_nejmcrmc_eval.py`: Creates NEJMCRMC dataset

Both scripts support:
- Batch processing with size 350
- Concurrent API calls for efficiency
- Progress tracking with tqdm
- Merging into eval_data.json with proper overwrite handling