"""
Clinical formatting ablation methods for analyzing components of reasoning extraction.
"""

from typing import Dict, Optional, List
import logging
from .clinical_formatting import transform_to_nejmcr_qa, transform_to_nejmcr_reason, transform_to_nejmcr_clean

def generate_reason_prompt(flags: Dict[str, bool]) -> str:
    """Generate the reasoning prompt based on ablation flags.
    
    Args:
        flags: Dictionary of boolean flags for each ablation component:
            - extraction: Just filter to reasoning
            - structure: Add step-by-step format
            - depth: Add comprehensiveness and knowledge
            - purity: Add exclusion of case report references
            - style: Add natural speaking style (full reason+clean)
    """
    base_prompt = """
Case report:
{cot}

Question:
{question}

Filter from the case report detailed reasoning for answering the question with "{answer}".
"""
    
    # Build up prompt incrementally based on flags
    requirements = []
    
    # Basic extraction
    if flags.get("extraction", False):
        requirements.append("1. Filter to only include reasoning relevant to answering the question.")
    
    # Add structure
    if flags.get("structure", False):
        requirements.append("2. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.")
    
    # Add depth
    if flags.get("depth", False):
        requirements.extend([
            "3. Provide maximally in-depth reasoning for every statement (i.e. include all pertinent positives, negatives, counterfactuals, alternatives, etc.).",
            "4. Rely primarily on biomedical and clinical knowledge from the case report.",
            "5. Include relevant reflection, backtracking, and self-validation present in the case report."
        ])
    
    # Add purity
    if flags.get("purity", False):
        requirements.extend([
            "6. Be written as if I am the doctor reasoning out loud having never seen the case report, ONLY having seen the facts in the QUESTION.",
            "7. NEVER cite the case report"
        ])
    
    # Add requirements if any exist
    if requirements:
        base_prompt += "\nThe reasoning must:\n" + "\n".join(requirements)
    
    base_prompt += "\n\nIMPORTANT: Start directly with the reasoning without any introduction or meta-text."
    
    return base_prompt

def generate_clean_prompt(flags: Dict[str, bool]) -> str:
    """Generate the cleaning prompt based on ablation flags."""
    base_prompt = """
Reasoning:
{cot}

Edit the reasoning with the following changes:
"""
    
    changes = []
    
    # Add purity-related cleaning
    if flags.get("purity", False):
        changes.append("1. Remove mentions of the case report. Change any language like \"the case report mentions that <biomedical/clinical rationale>\" --> to instead say \"<biomedical/clinical rationale>\".")
    
    # Add style-related cleaning
    if flags.get("style", False):
        changes.append("2. Edit the transitions in reasoning sound to sound more naturally dynamic, weaving in words like \"Oh\" \"Hmm\" \"Wait\"")
    
    if changes:
        base_prompt += "\n" + "\n".join(changes)
        base_prompt += "\n\nDo NOT edit anything else in the reasoning."
    
    base_prompt += "\n\nIMPORTANT: Start directly with the reasoning without any introduction or meta-text."
    
    return base_prompt

async def transform_to_nejmcr_reason_ablated(cot: str, question: str, answer: str, model_key: str, ablation_flags: Dict[str, bool]) -> str:
    """Transform case into detailed reasoning with specific ablations enabled/disabled.
    
    Args:
        cot: The case report/reasoning to transform
        question: The medical question
        answer: The correct diagnosis/answer
        model_key: The model to use for transformation
        ablation_flags: Dictionary of boolean flags for each ablation component
    """
    from utils.openai_utils import get_model_response
    import logging
    
    # Log input sizes
    logging.info(f"NEJMCR Reason Ablated transform inputs:")
    logging.info(f"CoT: {len(cot)} chars")
    logging.info(f"Question: {len(question)} chars")
    logging.info(f"Answer: {len(answer)} chars")
    logging.info(f"Ablation flags: {ablation_flags}")
    
    # Generate appropriate prompt based on flags
    prompt = generate_reason_prompt(ablation_flags)
    formatted_prompt = prompt.format(cot=cot, question=question, answer=answer)
    logging.info(f"Total prompt size: {len(formatted_prompt)} chars")
    
    result = await get_model_response(
        formatted_prompt,
        model=model_key,
        max_tokens=8192,
        raise_on_failure=True
    )
    
    if result is None:
        raise RuntimeError("get_model_response returned None despite raise_on_failure=True")
    
    # Only apply clean step if needed
    if ablation_flags.get("purity", False) or ablation_flags.get("style", False):
        clean_prompt = generate_clean_prompt(ablation_flags)
        result = await get_model_response(
            clean_prompt.format(cot=result),
            model=model_key,
            max_tokens=8192,
            raise_on_failure=True
        )
    
    logging.info(f"Successfully generated reasoning ({len(result)} chars)")
    return result

# Predefined ablation configurations
ABLATION_CONFIGS = {
    "extraction": {"extraction": True},
    "structure": {"extraction": True, "structure": True},
    "depth": {"extraction": True, "structure": True, "depth": True},
    "purity": {"extraction": True, "structure": True, "depth": True, "purity": True},
    "style": {"extraction": True, "structure": True, "depth": True, "purity": True, "style": True}
}