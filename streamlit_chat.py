#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import streamlit as st

@st.cache_resource
def load_model(model_path):
    """Load model and tokenizer only once"""
    print(f"Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=True
    )
    return model, tokenizer

def generate_response(model, tokenizer, user_input, thinking_placeholder, answer_placeholder):
    """Generate response with streaming output"""
    messages = [
        {"role": "user", "content": "You are a medical expert responding to the following query from the user: " + user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    def generate():
        with torch.no_grad():
            model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8192,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                streamer=streamer,
            )
    
    thread = Thread(target=generate)
    thread.start()
    
    # Process the streamed output
    started_thinking = False
    started_answer = False
    thinking_text = []
    answer_text = []
    
    for text in streamer:
        if not started_thinking and "think" in text.lower():
            started_thinking = True
            continue
        
        if started_thinking and not started_answer and ("answer" in text.lower() or "##" in text.lower()):
            started_answer = True
            continue
        
        if started_thinking:
            if not started_answer:
                thinking_text.append(text)
                thinking_placeholder.markdown("".join(thinking_text))
            else:
                answer_text.append(text.replace("##", "").replace("Final Response", "Answer: "))
                answer_placeholder.markdown("".join(answer_text))
    
    thread.join()

def main():
    st.set_page_config(page_title="med-s1", layout="wide")
    st.title("Learning Medical Reasoning")
    
    # Load models
    ft_model_path = "/share/pi/nigam/users/calebwin/hf_cache/ckpts/med_s1_/share/pi/nigam/data/med_s1k/s1_replication/med_s1k_formatted_bs8_lr1e-5_epoch5_wd1e-4_20250220_011621/checkpoint-450"
    huatuo_path = "FreedomIntelligence/HuatuoGPT-o1-7B"
    
    ft_model, ft_tokenizer = load_model(ft_model_path)
    huatuo_model, huatuo_tokenizer = load_model(huatuo_path)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("med-s1")
        st.caption("Llama-7B-Instruct trained on N=1,000")
    
    with col2:
        st.header("HuatuoGPT")
        st.caption("Llama-7B-Instruct trained on N=25,371")
    
    # Chat input
    user_input = st.chat_input("Ask a medical question...")
    
    if user_input:
        # Show user message in both columns
        with col1:
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant", avatar="🤔"):
                ft_thinking = st.empty()
            with st.chat_message("assistant", avatar="💡"):
                ft_answer = st.empty()
        
        with col2:
            st.chat_message("user").write(user_input)
            with st.chat_message("assistant", avatar="🤔"):
                huatuo_thinking = st.empty()
            with st.chat_message("assistant", avatar="💡"):
                huatuo_answer = st.empty()
        
        # Generate responses serially
        generate_response(ft_model, ft_tokenizer, user_input, ft_thinking, ft_answer)
        generate_response(huatuo_model, huatuo_tokenizer, user_input, huatuo_thinking, huatuo_answer)

if __name__ == "__main__":
    main()