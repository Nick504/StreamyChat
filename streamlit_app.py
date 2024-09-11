import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Qwen-2-7B model and tokenizer
@st.cache_resource  # Caching for efficiency
def load_model():
    model_name = "Qwen/Qwen-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI Setup
st.title("Vision Chatbot with Qwen-2-7B")
st.write("A basic chatbot interface using the Qwen-2-7B model.")

# Text input for user queries
user_input = st.text_input("You:", "")

if user_input:
    # Tokenize the input and generate a response
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    
    # Decode the generated tokens to get the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Display the response
    st.text_area("Bot:", value=response, height=100)