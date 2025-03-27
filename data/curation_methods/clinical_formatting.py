"""
Clinical formatting methods for transforming medical reasoning traces into various formats.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

async def transform_to_list(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a list of axioms."""
    prompt = """
You are an expert medical educator. Extract the key medical axioms and facts from this reasoning trace as a bullet-point list.

Each point should:
1. Be a single, clear medical fact or logical step
2. Start with a bullet point (-)
3. Be concise but complete
4. Preserve medical accuracy

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with the bullet points. Do not include any introduction or explanation.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=4096)
    
    # Ensure proper formatting
    if result:
        # Remove any text before first bullet point
        first_bullet = result.find('-')
        if first_bullet >= 0:
            result = result[first_bullet:]
        
        # Ensure each line starts with a bullet
        lines = result.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-'):
                line = f"- {line}"
            if line:
                formatted_lines.append(line)
        
        result = '\n'.join(formatted_lines)
    
    return result

async def transform_to_markdown(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a markdown document."""
    prompt = """
You are an expert medical educator. Transform this medical reasoning trace into a well-structured markdown document.

The document should:
1. Use appropriate markdown headings (##, ###)
2. Include relevant lists (ordered or unordered)
3. Use tables where appropriate
4. Maintain all medical accuracy
5. Be organized logically

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with markdown content. Do not include any meta-text or explanations.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=4096)
    
    # Basic markdown validation/cleanup
    if result:
        # Ensure sections start with ##
        if not result.startswith('#'):
            result = f"## Analysis\n{result}"
            
        # Fix common markdown formatting issues
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Remove excess newlines
        result = re.sub(r'(?<!\n)#', '\n#', result)  # Ensure headings start on new lines
        
    return result

async def transform_to_step_evidence(cot: str, model_key: str) -> str:
    """Transform reasoning trace into steps with clinical evidence."""
    prompt = """
You are an expert medical educator. Transform this reasoning trace into clear steps with supporting clinical evidence.

Each step should:
1. Have a clear title (## Step N: Title)
2. Include the main reasoning
3. Add "Evidence:" section with:
   - Similar clinical cases
   - Relevant research findings
   - Practice guidelines
4. Maintain medical accuracy

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with "## Step 1:" without any introduction.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=4096)
    
    # Ensure proper formatting
    if result:
        # Start with first step
        step1_match = re.search(r'## Step 1:', result)
        if step1_match:
            result = result[step1_match.start():]
            
        # Ensure evidence sections are properly formatted
        result = re.sub(r'(?<!#)#\s*Evidence:', '\n### Evidence:', result)
        
    return result

async def transform_to_note(cot: str, model_key: str) -> str:
    """
    Transform reasoning trace into the most appropriate clinical note format.
    First determines the best format (SOAP, SOAPIE, ISBAR, or POMR) based on the content,
    then transforms it accordingly.
    
    Args:
        cot: The reasoning trace to transform
        model_key: The model to use for transformation
    """
    # First determine the most appropriate format
    format_selection_prompt = """
You are an expert medical educator. Analyze this medical reasoning trace and determine the most appropriate clinical note format.

Choose from:
1. SOAP - Subjective, Objective, Assessment, Plan
2. SOAPIE - Subjective, Objective, Assessment, Plan, Implementation, Evaluation
3. ISBAR - Identification, Situation, Background, Assessment, Recommendation
4. POMR - Database, Problem List, Initial Plans, Progress Notes, Final Summary

Return ONLY the format name (SOAP, SOAPIE, ISBAR, or POMR) without any explanation.

Here's the reasoning trace:

{cot}
"""
    from utils.openai_utils import get_model_response
    
    format_type = await get_model_response(format_selection_prompt.format(cot=cot), model=model_key, max_tokens=10)
    format_type = format_type.strip().upper()
    
    # Validate format type
    if format_type not in ["SOAP", "SOAPIE", "ISBAR", "POMR"]:
        format_type = "SOAP"  # Default to SOAP if invalid response
    format_prompts = {
        "SOAP": """
Transform this medical reasoning into a SOAP note with these sections:
## Subjective
- Patient-reported symptoms and history

## Objective
- Observable findings and test results

## Assessment
- Clinical evaluation and diagnosis

## Plan
- Treatment strategy and follow-up
""",
        "SOAPIE": """
Transform this medical reasoning into a SOAPIE note with these sections:
## Subjective
- Patient-reported symptoms and history

## Objective
- Observable findings and test results

## Assessment
- Clinical evaluation and diagnosis

## Plan
- Treatment strategy

## Implementation
- How the plan was executed

## Evaluation
- Effectiveness of interventions
""",
        "ISBAR": """
Transform this medical reasoning into an ISBAR note with these sections:
## Identification
- Patient and provider identification

## Situation
- Current clinical situation

## Background
- Relevant clinical context

## Assessment
- Clinical assessment

## Recommendation
- Suggested actions
""",
        "POMR": """
Transform this medical reasoning into a POMR note with these sections:
## Database
- Comprehensive patient information

## Problem List
- All medical issues

## Initial Plans
- Interventions for each problem

## Progress Notes
- Ongoing documentation

## Final Summary
- Discharge or final status
"""
    }

    prompt = f"""
You are an expert medical educator. Transform this medical reasoning trace into a {format_type} clinical note.

The note should:
1. Follow this structure:
{format_prompts[format_type]}
2. Be comprehensive but concise
3. Maintain all medical accuracy
4. Use appropriate medical terminology

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with the first section heading (##). Do not include any introduction.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=4096)
    
    # Ensure proper formatting
    if result:
        # Start with first section
        first_section = re.search(r'##\s+\w+', result)
        if first_section:
            result = result[first_section.start():]
            
        # Fix section formatting
        result = re.sub(r'(?<!#)#(?!#)', '##', result)  # Ensure consistent heading level
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Remove excess newlines
        
    return result