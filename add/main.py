import streamlit as st
from inference import load_model, generate_response
from prompts import build_simplification_prompt

st.title("Adaptive Discharge AI")

@st.cache_resource
def initialize():
    return load_model()

tokenizer, model = initialize()

discharge_text = st.text_area("Paste discharge summary:")

literacy_level = st.selectbox(
    "Select patient literacy level:",
    ["6th grade", "10th grade", "medical professional"]
)

if st.button("Simplify"):
    prompt = build_simplification_prompt(discharge_text, literacy_level)
    result = generate_response(tokenizer, model, prompt)
    st.subheader("Simplified Instructions")
    st.write(result)
