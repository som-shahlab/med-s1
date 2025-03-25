def print_indented(text: str):
    """Prints each line of the string with one tab indentation."""
    for line in text.split('\n'):
        print(f'\t{line}')

def postprocess_output(pred: str) -> str:
    """Postprocess the output of the model."""
    pred = pred.replace("</s>", "")
    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def get_query_prompt(args, experiment=None):
    """Get the appropriate query prompt based on args and experiment config."""
    # Check if experiment config has a prompting approach set to "step"
    step_prompt = False
    if experiment and "config" in experiment:
        step_prompt = experiment["config"].get("prompting", "") == "step"
    
    if args.test_time_scaling or args.strict_prompt:
        base_prompt = "Please answer the following multiple-choice question, ensuring your response concludes with the correct option in the format: 'The answer is BLANK' where BLANK is the correct option. For example, if the correct answer is A, your response should be 'The answer is A.'."
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"
    else:
        base_prompt = "Please answer the following multiple-choice question:"
        if step_prompt:
            return f"{base_prompt}\n{{question}}\n{{option_str}}\n\nLet's think step by step."
        else:
            return f"{base_prompt}\n{{question}}\n{{option_str}}"