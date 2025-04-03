import torch
import transformers
import threading

def format_chat_prompt(system_content, messages):
    """
    Apply the chat template to format a conversation.
    
    The template produces a prompt that looks like:
    
      <|start_header_id|>system<|end_header_id|>
      [system content]
      <|eot_id|>
      <|start_header_id|>user<|end_header_id|>
      [user content]
      <|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>
    
    You can extend this for additional roles as needed.
    """
    prompt = ""
    # Start with system header
    prompt += "<|start_header_id|>system<|end_header_id|>\n"
    prompt += system_content.strip() + "\n"
    prompt += "<|eot_id|>\n"
    # Add each message (skipping system messages if already in the header)
    for i, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        is_last = (i == len(messages) - 1)
        # For a chat, we assume roles like "user" and "assistant" (or others)
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n"
        prompt += content + "\n"
        prompt += "<|eot_id|>\n"
        # If this is the last message, add an assistant header as the output start
        if is_last:
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

# ----- Setup Model and Tokenizer -----
model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

# Load tokenizer and set pad token appropriately
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

# ----- Define Chat Messages and Format the Prompt -----
system_message = "detailed thinking on"
# Our chat conversation: system message is provided in the header, and here we have one user message.
messages = [
    {"role": "user", "content": "Solve x*(sin(x)+2)=0"}
]

# Apply the chat template formatting
prompt = format_chat_prompt(system_message, messages)
print("Formatted prompt:\n", prompt)

# ----- Tokenize the Prompt -----
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# ----- Set Up Streaming Generation -----
# TextIteratorStreamer will yield text chunks as tokens are generated.
streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=32768,
    do_sample=True,      # Use sampling (or set to False for deterministic output)
    temperature=0.6,
    top_p=0.95,
    streamer=streamer,
)

# Run generation in a background thread so that we can stream the output
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# ----- Stream and Print Tokens as They Are Generated -----
for new_text in streamer:
    print(new_text, end="", flush=True)

thread.join()

