"""
Clinical formatting methods for transforming chain of thought reasonings into various formats.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

async def transform_to_list(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a list of axioms."""
    prompt = """
You are an expert medical educator. Extract the key axioms and facts from this reasoning trace as a bullet-point list.

Each point should:
1. Be a single, clear fact or logical step
2. Start with a bullet point (-)

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with the bullet points. Do not include any text before the first bullet point.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
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
You are an expert medical educator. Transform this chain of thought reasoning into a well-structured markdown document.

The document should:
1. Use appropriate markdown headings (##, ###)
2. Include relevant hierarchical lists (ordered or unordered)
3. Use tables where appropriate

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with markdown content. Do not include any meta-text or explanations.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
    return result

async def transform_to_step_evidence(cot: str, model_key: str) -> str:
    """Transform reasoning trace into steps with clinical evidence."""
    prompt = """
You are an expert medical educator. Transform this reasoning trace into clear steps with supporting clinical evidence.

Each step should:
1. Have a clear title (## Step N: Title)
2. Include the main reasoning
3. Add evidence section (**Evidence:**) with either:
   - Similar case
   - Relevant literature
4. Maintain medical accuracy

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with "## Step 1:" without any introduction.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
    # Ensure proper formatting
    if result:
        # Start with first step
        step1_match = re.search(r'## Step 1:', result)
        if step1_match:
            result = result[step1_match.start():]
            
        # Ensure evidence sections are properly formatted
        result = re.sub(r'(?<!#)#\s*Evidence:', '\n### Evidence:', result)
        
    return result

async def transform_to_steps(cot: str, model_key: str, extract_type: str = "step") -> str:
    """Transform reasoning trace into steps or a 1-sentence summary."""
    if extract_type == "step":
        prompt = """
You are an expert medical educator. Your task is to transform the following chain of thought reasoning into a clear, step-by-step format.

Each step should:
1. Be numbered and have a clear title (e.g., "## Step 1: Assess the patient's condition")
2. Include all content of the original reasoning
3. Be organized in a logical sequence
4. Maintain all medical accuracy and details from the original text

Here's the chain of thought reasoning to transform:

{cot}

IMPORTANT: Your response must start directly with "## Step 1:" without any introduction or preamble. Do not include any text before the first step.
"""
    else:  # 1-sentence
        prompt = """
You are an expert medical educator. Your task is to transform the following chain of thought reasoning into a single, comprehensive sentence.

The sentence should:
1. Capture the key reasoning steps and logic from the original text
2. Be concise but complete, covering the main diagnostic process
3. Maintain all medical accuracy from the original text
4. Be no longer than 80 words

Here's the chain of thought reasoning to transform:

{cot}

IMPORTANT: Your response must be EXACTLY ONE SENTENCE. Do not include any introduction or multiple sentences. Start directly with the sentence and end with a period. Do not include any text before or after the sentence.
"""

    from utils.openai_utils import get_model_response
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)

    # Post-process the result based on extraction type
    if result:
        if extract_type == "step":
            # Check if "## Step 1:" exists in the string and clip to that point
            step1_match = re.search(r'## Step 1:', result)
            if step1_match:
                result = result[step1_match.start():]
            else:
                logging.warning(f"Step extraction did not produce expected format. Raw result: {result[:100]}...")
        else:  # 1-sentence
            # Basic cleanup for the 1-sentence result
            result = result.strip()
            
            # Remove any extra newlines or multiple spaces
            result = re.sub(r'\s+', ' ', result)
            
            # If it's too long, truncate with ellipsis
            if len(result) > 700:
                result = result[:697] + "..."
                
            # Ensure it ends with a period
            if not result.endswith('.'):
                result = result + '.'

    return result

async def transform_to_decision_tree(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a decision tree format."""
    prompt = """
You are an expert medical educator. Transform this chain of thought reasoning into a clear decision tree format.

The decision tree should:
1. Show the key decision points and logical flow
2. Use indentation to indicate hierarchy
3. Use -> to show decision paths
4. Indicate the decisions taken

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with the decision tree. Do not include any introduction or explanation.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
    # Ensure proper formatting
    if result:
        # Remove any text before first decision point
        first_arrow = result.find('->')
        if first_arrow >= 0:
            # Find start of line containing first arrow
            line_start = result.rfind('\n', 0, first_arrow)
            if line_start >= 0:
                result = result[line_start+1:]
            else:
                result = result
                
        # Ensure consistent arrow formatting
        result = re.sub(r'\s*-+>\s*', ' -> ', result)
        
    return result

async def transform_to_qa(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a Q&A format."""
    prompt = """
You are an expert medical educator. Transform this chain of thought reasoning into a sequence of Questions and Answers.

Each Q&A pair should:
1. Start with "Q:" for questions and "A:" for answers
2. Be on separate lines
3. Cover key points from the reasoning
5. Follow a logical progression

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with "Q:" without any introduction. Each Q&A should be on its own line.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
    # Ensure proper formatting
    if result:
        # Start with first Q:
        first_q = result.find('Q:')
        if first_q >= 0:
            result = result[first_q:]
            
        # Ensure consistent Q/A formatting
        result = re.sub(r'\n\s*Q:\s*', '\nQ: ', result)
        result = re.sub(r'\n\s*A:\s*', '\nA: ', result)
        
    return result

async def transform_to_socratic(cot: str, model_key: str) -> str:
    """Transform reasoning trace into a Socratic dialogue."""
    prompt = """
You are an expert medical educator. Transform this chain of thought reasoning into a Socratic dialogue between domain experts.

The dialogue should:
1. Use labeled speakers (e.g., "Cardiologist:", "Neurologist:", etc.)
2. Put each speaker's contribution on a new line
3. Use domain-appropriate experts withe fine domain granularity
4. Follow a logical progression through the reasoning
5. Use serious and professional language

Here's the reasoning trace:

{cot}

IMPORTANT: Start directly with the first speaker's line. Do not include any introduction.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(cot=cot), model=model_key, max_tokens=8192)
    
    # Ensure proper formatting
    if result:
        # Find first speaker
        first_colon = result.find(':')
        if first_colon >= 0:
            # Find start of line containing first speaker
            line_start = result.rfind('\n', 0, first_colon)
            if line_start >= 0:
                result = result[line_start+1:]
            else:
                result = result
                
        # Ensure consistent speaker formatting
        result = re.sub(r'\n\s*([^:\n]+):\s*', r'\n\1: ', result)
        
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
You are an expert medical educator. Analyze this chain of thought reasoning and determine the most appropriate note format.

Choose from:
1. SOAP - Subjective, Objective, Assessment, Plan
2. SOAPIE - Subjective, Objective, Assessment, Plan, Implementation, Evaluation
3. ISBAR - Identification, Situation, Background, Assessment, Recommendation
4. POMR - Database, Problem List, Initial Plans, Progress Notes, Final Summary

Return ONLY the format name (SOAP, SOAPIE, ISBAR, or POMR) without any text before the note.

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
Transform this chain of thought reasoning into a SOAP note with these sections:
## Subjective
- Reported symptoms and history

## Objective
- Observable findings and test results

## Assessment
- Evaluation and diagnosis

## Plan
- Treatment strategy and follow-up
""",
        "SOAPIE": """
Transform this chain of thought reasoning into a SOAPIE note with these sections:
## Subjective
- Reported symptoms and history

## Objective
- Observable findings and test results

## Assessment
- Evaluation and diagnosis

## Plan
- Treatment strategy

## Implementation
- How the plan was executed

## Evaluation
- Effectiveness of interventions
""",
        "ISBAR": """
Transform this chain of thought reasoning into an ISBAR note with these sections:
## Identification
- Who you are, your role, where you are and why you are
communicating

## Situation
- What is happening at the moment

## Background
- What are the issues that led up to this situation

## Assessment
- What do you believe the problem is

## Recommendation
- What should be done to correct this situation
""",
        "POMR": """
Transform this reasoning into a POMR note with these sections:
## Database
- What data was available

## Problem List
- What problems were identified

## Initial Plans
- What plans or interventions were proposed

## Progress Notes
- Any ongoing or future steps

## Final Summary
- Final status
"""
    }

    prompt = """
You are an expert medical educator. Transform this chain of thought reasoning into a {format_type} clinical note.

Follow this structure:
{format_structure}

Here's the chain of thought reasoning:

{cot}

IMPORTANT: Start directly with the first section heading (##). Do not include any introduction.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(prompt.format(
        format_type=format_type,
        format_structure=format_prompts[format_type],
        cot=cot
    ), model=model_key, max_tokens=8192)
    
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

async def transform_to_cot(question: str, answer: str, model_key: str) -> str:
    """Transform question and answer into Chain of Thought reasoning."""
    prompt = """
{question}
{answer}
Please respond to the above question using the Chain of Thought (CoT) reasoning method to reach the answer.

The Chain of Thought should resemble a human-like, intuitive natural thinking process. It should:
1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions. Use casual and natural language for
transitions or validations, such as "hmm," "oh," "also," or "wait."
3. Expand the content, making the reasoning richer, more detailed, and logically clear while still being
conversational and intuitive.

IMPORTANT: Start directly with your reasoning without any introduction or meta-text. Do not repeat the question or answer at the start.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(
        prompt.format(question=question, answer=answer),
        model=model_key,
        max_tokens=8192
    )
    
    if result:
        # Remove any meta-text or explanations at the start
        lines = result.split('\n')
        content_lines = []
        started = False
        for line in lines:
            # Skip empty lines at start
            if not started and not line.strip():
                continue
            # Skip lines that look like meta-text
            if not started and (
                line.lower().startswith(('here', 'let', 'i will', 'first', 'okay', 'now', 'using', 'let\'s'))
                or 'chain of thought' in line.lower()
                or 'reasoning' in line.lower()
            ):
                continue
            started = True
            content_lines.append(line)
        
        result = '\n'.join(content_lines)
    
    return result