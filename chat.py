#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import sys

def main():
    # Model path
    model_path = "/share/pi/nigam/users/calebwin/hf_cache/ckpts/med_s1_/share/pi/nigam/data/med_s1k/s1_replication/med_s1k_formatted_bs8_lr1e-5_epoch5_wd1e-4_20250220_011621/checkpoint-450"
    
    print("Loading med-s1...")
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=True
    )
    
    print("\nAsk med-s1 (Ctrl+C to exit)\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            print()  # Add newline after user input
            
            # Create chat format
            dialog = [{"role": "user", "content": user_input}]
            prompt = tokenizer.apply_chat_template(dialog, tokenize=False)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
            
            # Set up streamer
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
            
            # Create generation kwargs
            generation_kwargs = dict(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
            
            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Initialize flags and buffers
            started_thinking = False
            started_answer = False
            current_section = []
            
            # Process the streamed output
            for text in streamer:
                # Skip until we find "think"
                if not started_thinking and "think" in text.lower():
                    print("Thinking: ", end="", flush=True)
                    started_thinking = True
                    continue
                
                # Handle transition to answer
                if started_thinking and not started_answer and "answer" in text.lower():
                    print("\n", end="", flush=True)
                    started_answer = True
                    continue
                
                # Print content if we're in a section
                if started_thinking:
                    print(text, end="", flush=True)
            
            print("\n")  # Add final newline
            thread.join()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()