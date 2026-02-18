import streamlit as st
from inference import load_model, generate_response, _passes_safety
from structure import extract_sections
from litteracy import readability_scores

st.title("Adaptive Discharge AI")

@st.cache_resource
def initialize():
    return load_model()

tokenizer, model = initialize()

discharge_text = st.text_area("Paste discharge summary:")

profile = st.selectbox(
    "Select patient profile:",
    ["low_literacy", "elderly", "standard", "post_surgery", "cardiac"]
)

if st.button("Simplify"):
    if not discharge_text.strip():
        st.warning("Please paste a discharge summary text.")
    else:
        sections = extract_sections(discharge_text)
        result = generate_response(tokenizer, model, sections, profile)
        st.subheader("Simplified Instructions")
        st.write(result)

        # --- Metrics & Safety Indicators ---
        with st.expander("Quality & Safety Checks"):
            before_scores = readability_scores(discharge_text)
            after_scores = readability_scores(result)
            st.markdown("Readability (lower is easier):")
            c1, c2, c3 = st.columns(3)
            c1.metric("FKGL (before)", f"{before_scores['fkgl']:.2f}" if before_scores['fkgl'] >= 0 else "n/a")
            c2.metric("FKGL (after)", f"{after_scores['fkgl']:.2f}" if after_scores['fkgl'] >= 0 else "n/a")
            c3.metric("Length ratio", f"{(len(result)+1)/(len(discharge_text)+1):.2f}")

            original_concat = "\n".join(v for v in sections.values() if v)
            numbers_ok = _passes_safety(original_concat, result)
            st.write(f"Numbers/critical words preserved: {'✅' if numbers_ok else '❌'}")
