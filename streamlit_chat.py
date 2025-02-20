#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import streamlit as st
from queue import Queue

def initialize_model():
    model_path = "/share/pi/nigam/users/calebwin/hf_cache/ckpts/med_s1_/share/pi/nigam/data/med_s1k/s1_replication/med_s1k_formatted_bs8_lr1e-5_epoch5_wd1e-4_20250220_011621/checkpoint-450"
    
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
    
    return model, tokenizer

def main():
    st.set_page_config(page_title="med-s1", layout="wide")
    st.title("med-s1")
    
    # Initialize session state
    if "model" not in st.session_state:
        with st.spinner("Loading Llama-7b trained on 1,000 curated medical reasoning traces..."):
            model, tokenizer = initialize_model()
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
    
    # Chat input
    user_input = st.chat_input("Ask a medical question...")
    
    if user_input:
        # Show user message
        st.chat_message("user").write(user_input)
        
        # Create chat format
        dialog = [{"role": "user", "content": user_input}]
        prompt = st.session_state.tokenizer.apply_chat_template(dialog, tokenize=False)
        
        # Tokenize input
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt", padding=True).to(st.session_state.model.device)
        
        # Set up streamer
        streamer = TextIteratorStreamer(st.session_state.tokenizer, skip_special_tokens=True)
        
        # Create generation kwargs
        generation_kwargs = dict(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            streamer=streamer,
        )
        
        # Start generation in a separate thread
        thread = Thread(target=st.session_state.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Create message containers
        with st.chat_message("assistant", avatar="🤔"):
            thinking_placeholder = st.empty()
        
        with st.chat_message("assistant", avatar="💡"):
            answer_placeholder = st.empty()
        
        # Initialize flags and buffers
        started_thinking = False
        started_answer = False
        thinking_text = []
        answer_text = []
        
        # Process the streamed output
        for text in streamer:
            # Skip until we find "think"
            if not started_thinking and "think" in text.lower():
                started_thinking = True
                continue
            
            # Handle transition to answer
            if started_thinking and not started_answer and "answer" in text.lower():
                started_answer = True
                continue
            
            # Accumulate text in appropriate section
            if started_thinking:
                if not started_answer:
                    thinking_text.append(text)
                    thinking_placeholder.write("".join(thinking_text))
                else:
                    answer_text.append(text)
                    answer_placeholder.write("".join(answer_text))
        
        thread.join()

if __name__ == "__main__":
    main()