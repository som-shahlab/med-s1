"""
Clinical formatting methods for transforming chain of thought reasonings into various formats.
"""

import re
import os

# Enable debug logging for Gemini transformation
from openai import OpenAI
TRANSFORM_DEBUG = True  # Debug logging for transformations
import logging
from typing import Dict, Optional, List, Tuple
import asyncio
from google import genai

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

async def transform_to_nejmcr_steps(cot: str, model_key: str, answer: str) -> str:
    """Transform reasoning trace into NEJM case report style steps with additional cleanup."""
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
    from utils.openai_utils import get_model_response
    
    # First pass to get initial step-by-step format
    result = await get_model_response(prompt.format(cot=cot, answer=answer), model=model_key, max_tokens=8192)
    
    if result:
        # Start with first step
        step1_match = re.search(r'## Step 1:', result)
        if step1_match:
            result = result[step1_match.start():]
        
        # Second pass to ensure proper ordering and cleanup
        cleanup_prompt = """
Review and improve these reasoning steps to ensure they:
1. Progress in a logical order towards the final diagnosis: {answer}
2. Are free of references to doctors, tables, figures
3. Focus only on the diagnostic process (exclude post-diagnosis actions taken, include discussion rationalizing the diagnosis before the diagnosis is made)
4. Resemble a human-like, intuitive natural thinking process (not a passive-voice post-hoc report)

Reasoning steps:
{steps}

IMPORTANT: Start directly with "## Step 1:".
"""
        result = await get_model_response(cleanup_prompt.format(steps=result, answer=answer), model=model_key, max_tokens=8192)
        
        # Ensure it starts with Step 1
        if result:
            step1_match = re.search(r'## Step 1:', result)
            if step1_match:
                result = result[step1_match.start():]
    
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

async def transform_to_gemini(question: str, model_key: str, answer: str) -> str:
    """Transform using either Gemini or DeepSeek model with multiple attempts to match answer.
    
    Args:
        question: The medical question to analyze
        model_key: The model key to use (determines which implementation to use)
        answer: The correct diagnosis/answer to compare against
    
    Returns:
        The best matching reasoning trace from the attempts
    """
    if model_key == "deepseek-reasoner":
        return await transform_to_gemini_with_deepseek(question, model_key, answer)
    else:
        return await transform_to_gemini_with_gemini(question, model_key, answer)

async def transform_to_gemini_with_gemini(question: str, model_key: str, answer: str) -> str:
    """Transform using Gemini model with get_model_response."""
    from utils.openai_utils import get_model_response
    
    if TRANSFORM_DEBUG:
        logging.info("Starting Gemini transformation")
        logging.info(f"Question: {question}")
        logging.info(f"Correct answer: {answer}")
    
    # Run 3 times
    responses = []
    response_answers = []
    
    # Prompt for both reasoning and diagnosis
    prompt = f"""
{question}

Provide your detailed medical reasoning and final diagnosis. Include:
1. Your step-by-step thought process
2. Key findings and their significance
3. Differential diagnoses considered
4. Final diagnosis

IMPORTANT: End your response with "Final Diagnosis:" followed by your conclusion.
"""
    
    for i in range(3):
        try:
            if TRANSFORM_DEBUG:
                logging.info(f"\nAttempt {i+1}/3")
            
            # Get response
            result = await get_model_response(prompt, model=model_key, max_tokens=8192)
            
            # Try to extract thinking and answer using various patterns
            thinking = None
            response_answer = None
            
            # Common patterns for final diagnosis
            patterns = [
                "Final Diagnosis:",
                "Final diagnosis:",
                "Therefore, the diagnosis is",
                "Thus, the diagnosis is",
                "The diagnosis is",
                "In conclusion,"
            ]
            
            # Try each pattern
            for pattern in patterns:
                if pattern in result:
                    parts = result.split(pattern)
                    if len(parts) >= 2:
                        thinking = parts[0].strip()
                        response_answer = pattern + parts[1].strip()
                        break
            
            # If no pattern matched, try to split on the last sentence
            if thinking is None:
                sentences = result.split('.')
                if len(sentences) > 1:
                    thinking = '.'.join(sentences[:-1]).strip()
                    response_answer = sentences[-1].strip()
            
            # If we found both parts, store them
            if thinking and response_answer:
                responses.append(thinking)
                response_answers.append(response_answer)
                
                if TRANSFORM_DEBUG:
                    logging.info(f"Generated thinking ({len(thinking)} chars)")
                    logging.info(f"Generated answer: {response_answer}")
                    logging.info("Response structure:")
                    logging.info(f"- Thinking: {thinking[:100]}...")
                    logging.info(f"- Answer: {response_answer}")
            
        except Exception as e:
            logging.error(f"API error on attempt {i+1}: {str(e)}")
            if TRANSFORM_DEBUG:
                logging.error(f"Full error: {str(e)}")
            continue
    
    # If we have responses, compare them using Gemini
    if responses:
        # First check for exact matches
        gemini_client = genai.Client(
            vertexai=True,
            project="som-nero-phi-nigam-starr",
            location="us-central1"
        )
        
        # Label responses for final comparison
        labeled_responses = []
        exact_matches = []
        
        for i, (thinking, response_answer) in enumerate(zip(responses, response_answers)):
            try:
                if TRANSFORM_DEBUG:
                    logging.info(f"\nChecking response {i+1}")
                    logging.info(f"Response answer: {response_answer}")
                
                # Check if diagnoses are identical
                comparison_prompt = f"""Are these two diagnoses identical? Answer yes or no.

Diagnosis 1: {response_answer}
Diagnosis 2: {answer}"""
                
                comparison = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=comparison_prompt
                )
                is_identical = comparison.candidates[0].content.parts[0].text.strip().lower()
                
                if TRANSFORM_DEBUG:
                    logging.info(f"Are diagnoses identical? {is_identical}")
                
                if is_identical == "yes":
                    exact_matches.append(thinking)
                
                # Label each response for final comparison
                labeled_responses.append((chr(65 + i), thinking, response_answer))
                
            except Exception as e:
                logging.error(f"Error comparing response {i}: {str(e)}")
                continue
        
        # If we found exact matches, use the first one
        if exact_matches:
            if TRANSFORM_DEBUG:
                logging.info("Found exact match(es)")
            return exact_matches[0]
        
        # Otherwise, ask which response is most similar
        try:
            if TRANSFORM_DEBUG:
                logging.info("\nFinding most similar response")
            
            comparison_prompt = f"""Consider these diagnoses:

{chr(10).join(f"{label}. {resp_answer}" for label, _, resp_answer in labeled_responses)}

Correct diagnosis: {answer}

Which lettered diagnosis (A, B, or C) is most similar to the correct diagnosis? Answer with just the letter."""
            
            comparison = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=comparison_prompt
            )
            best_letter = comparison.candidates[0].content.parts[0].text.strip()
            
            if TRANSFORM_DEBUG:
                logging.info(f"Most similar response: {best_letter}")
            
            # Find the thinking that corresponds to the best letter
            for label, thinking, _ in labeled_responses:
                if label == best_letter:
                    return thinking
            
        except Exception as e:
            logging.error(f"Error in final comparison: {str(e)}")
        
        # If all else fails, return the first response
        if TRANSFORM_DEBUG:
            logging.info("Using first response as fallback")
        return responses[0]
    
    return question

async def transform_to_gemini_with_deepseek(question: str, model_key: str, answer: str) -> str:
    """Transform using DeepSeek model."""
    client = OpenAI(api_key="sk-8f99cc339b994552bd046d5b35ce0dd9", base_url="https://api.deepseek.com")
    
    if TRANSFORM_DEBUG:
        logging.info("Starting DeepSeek transformation")
        logging.info(f"Question: {question}")
        logging.info(f"Correct answer: {answer}")
    
    # Run 3 times
    responses = []
    response_answers = []
    
    for i in range(3):
        try:
            if TRANSFORM_DEBUG:
                logging.info(f"\nAttempt {i+1}/3")
            
            # Call DeepSeek with just the question
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": question}]
            )
            
            # Get thinking and answer directly from response
            thinking = response.choices[0].message.reasoning_content
            response_answer = response.choices[0].message.content
            
            responses.append(thinking)
            response_answers.append(response_answer)
            
            if TRANSFORM_DEBUG:
                logging.info(f"Generated thinking ({len(thinking)} chars)")
                logging.info(f"Generated answer: {response_answer}")
                logging.info("Response structure:")
                logging.info(f"- Thinking: {thinking[:100]}...")
                logging.info(f"- Answer: {response_answer}")
            
        except Exception as e:
            logging.error(f"DeepSeek API error on attempt {i+1}: {str(e)}")
            if TRANSFORM_DEBUG:
                logging.error(f"Full error: {str(e)}")
            continue
    
    # If we have responses, compare them using Gemini
    if responses:
        # First check for exact matches
        gemini_client = genai.Client(
            vertexai=True,
            project="som-nero-phi-nigam-starr",
            location="us-central1"
        )
        
        # Label responses for final comparison
        labeled_responses = []
        exact_matches = []
        
        for i, (thinking, response_answer) in enumerate(zip(responses, response_answers)):
            try:
                if TRANSFORM_DEBUG:
                    logging.info(f"\nChecking response {i+1}")
                    logging.info(f"Response answer: {response_answer}")
                
                # Check if diagnoses are identical
                comparison_prompt = f"""Are these two diagnoses identical? Answer yes or no.

Diagnosis 1: {response_answer}
Diagnosis 2: {answer}"""
                
                comparison = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=comparison_prompt
                )
                is_identical = comparison.candidates[0].content.parts[0].text.strip().lower()
                
                if TRANSFORM_DEBUG:
                    logging.info(f"Are diagnoses identical? {is_identical}")
                
                if is_identical == "yes":
                    exact_matches.append(thinking)
                
                # Label each response for final comparison
                labeled_responses.append((chr(65 + i), thinking, response_answer))
                
            except Exception as e:
                logging.error(f"Error comparing response {i}: {str(e)}")
                continue
        
        # If we found exact matches, use the first one
        if exact_matches:
            if TRANSFORM_DEBUG:
                logging.info("Found exact match(es)")
            return exact_matches[0]
        
        # Otherwise, ask which response is most similar
        try:
            if TRANSFORM_DEBUG:
                logging.info("\nFinding most similar response")
            
            comparison_prompt = f"""Consider these diagnoses:

{chr(10).join(f"{label}. {resp_answer}" for label, _, resp_answer in labeled_responses)}

Correct diagnosis: {answer}

Which lettered diagnosis (A, B, or C) is most similar to the correct diagnosis? Answer with just the letter."""
            
            comparison = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=comparison_prompt
            )
            best_letter = comparison.candidates[0].content.parts[0].text.strip()
            
            if TRANSFORM_DEBUG:
                logging.info(f"Most similar response: {best_letter}")
            
            # Find the thinking that corresponds to the best letter
            for label, thinking, _ in labeled_responses:
                if label == best_letter:
                    return thinking
            
        except Exception as e:
            logging.error(f"Error in final comparison: {str(e)}")
        
        # If all else fails, return the first response
        if TRANSFORM_DEBUG:
            logging.info("Using first response as fallback")
        return responses[0]
    
    return question

async def transform_to_nejmcr_qa(question: str, cot: str, answer: str, model_key: str) -> str:
    """Transform case into a challenging medical exam question."""
    prompt = """
Case report - detailed vignette:
{question}

Case report - reasoning:
{cot}

Case report - diagnosis:
{answer}
                 
Filter from the case report a challenging medical exam question and answer.
                 
The exam question must:
1. Describe a clinical vignette and ask a question about clinical decision making (e.g. diagnosis, treatment planning, etc.).
2. Have a specific answer but not display multiple choices.
3. Be extremely challenging but definitely solveable by a very experienced licensed medical professional who has only read the extracted question.
4. Be entirely based on facts in the case report.
5. Contain all details from the case report required to reason comprehensively to arrive at the answer.
6. Be solvable with only the information in the question.

Respond with the question and answer in the following format:
Question: <question>
Answer: <answer>

IMPORTANT: Start directly with Question: without any introduction or meta-text.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(
        prompt.format(question=question, cot=cot, answer=answer),
        model=model_key,
        max_tokens=8192
    )
    
    return result

async def transform_to_nejmcr_reason(cot: str, question: str, answer: str, model_key: str) -> str:
    """Transform case into detailed reasoning for the diagnosis."""
    prompt = """
Case report:
{cot}

Question:
{question}

Filter from the case report detailed reasoning for answering the question with "{answer}".

The reasoning must:
1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Provide maximally in-depth reasoning for every statement (i.e. include all pertinent positives, negatives, counterfactuals, alternatives, etc.).
3. Be a plausible reasoning based ONLY on facts provided in the QUESTION
4. Rely primarily on biomedical and clinical knowledge from the case report.
5. Be written as if I am the doctor reasoning out loud having never seen the case report, only having seen the question.
6. NEVER mention the case report

IMPORTANT: The reasoning must NEVER cite the case report or say something like "the case report mentions that <biomedical/clinical rationale>" --> instead say "<biomedical/clinical rationale>".

IMPORTANT: Start directly with the reasoning without any introduction or meta-text.
"""
    from utils.openai_utils import get_model_response
    import logging
    
    # Log input sizes to help diagnose potential issues
    logging.info(f"NEJMCR Reason transform inputs:")
    logging.info(f"CoT: {len(cot)} chars")
    logging.info(f"Question: {len(question)} chars")
    logging.info(f"Answer: {len(answer)} chars")
    
    formatted_prompt = prompt.format(cot=cot, question=question, answer=answer)
    logging.info(f"Total prompt size: {len(formatted_prompt)} chars")
    
    # Let API errors propagate up - they should be caught and logged at a higher level
    result = await get_model_response(
        formatted_prompt,
        model=model_key,
        max_tokens=8192,
        raise_on_failure=True  # Explicitly set to raise on failure
    )
    
    # This should never happen since raise_on_failure=True
    if result is None:
        raise RuntimeError("get_model_response returned None despite raise_on_failure=True")
    
    logging.info(f"Successfully generated reasoning ({len(result)} chars)")
    return result

async def transform_to_nejmcr_clean(cot: str, model_key: str) -> str:
    """Clean reasoning to remove case report mentions."""
    prompt = """
Reasoning:
{cot}

Edit the reasoning with the following changes:
1. Remove mentions of the case report. Change any language like "the case report mentions that <biomedical/clinical rationale>" --> to instead say "<biomedical/clinical rationale>".

Do NOT edit anything else in the reasoning.

IMPORTANT: Start directly with the reasoning without any introduction or meta-text.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(
        prompt.format(cot=cot),
        model=model_key,
        max_tokens=8192
    )
    
    return result

async def transform_to_nejmcr_transform(cot: str, model_key: str, answer: str) -> str:
    """Transform using the NEJMCR transformation prompt."""
    prompt = """
Extract from the following case report the detailed reasoning for diagnosing the patient with {answer}.

The reasoning trace should resemble a human-like, intuitive natural thinking process. It should:
1. Be presented as step-by-step reasoning, with each thought on a new line separated by a line break.
2. Avoid structured titles or formatting, focusing on natural transitions.
3. Not reference doctor names, tables, figures.
4. Avoid statements not related to the diagnostic reasoning process.
5. Provide maximally in-depth reasoning for every statement (i.e. rule-out, thought, etc.).
6. Be written in the first person, as if I am the doctor reasoning out loud.

Case report:
{cot}

IMPORTANT: Start directly with your reasoning without any introduction or meta-text.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(
        prompt.format(cot=cot, answer=answer),
        model=model_key,
        max_tokens=8192
    )
    
    return result

async def transform_to_gemini_nejmcr(question: str, model_key: str, answer: str, cot: str) -> str:
    """Transform using model with NEJMCR-style enhancement.
    
    Args:
        question: The medical question to analyze
        model_key: The model key to use
        answer: The correct diagnosis/answer
        cot: The original case report/CoT to use for enhancement
    
    Returns:
        Enhanced reasoning combining initial response with case report analysis
    """
    if TRANSFORM_DEBUG:
        logging.info("Starting NEJMCR transformation")
        logging.info("Step 1: Getting initial response")
    
    # First get best response using the question
    initial_response = await transform_to_gemini(question, model_key, answer)
    
    if TRANSFORM_DEBUG:
        logging.info("Step 2: Enhancing with case report analysis")
        logging.info(f"Initial response length: {len(initial_response)}")
        logging.info(f"Original CoT length: {len(cot)}")
    
    # Then enhance it with case report analysis
    enhancement_prompt = f"""
Review this diagnostic reasoning and the original case report. Your task is to:
1. Fill in any missing reasoning steps
2. Add items that should be ruled out
3. Correct any incorrect reasoning
4. Ensure the logic flows naturally to the diagnosis: {answer}

Original case report:
{cot}

Current reasoning:
{initial_response}

IMPORTANT: Provide the complete enhanced reasoning, starting directly with the first step.
"""
    from utils.openai_utils import get_model_response
    
    result = await get_model_response(
        enhancement_prompt,
        model=model_key,
        max_tokens=8192
    )
    
    if TRANSFORM_DEBUG:
        logging.info("Enhancement complete")
        logging.info(f"Final result length: {len(result) if result else 0}")
    
    return result