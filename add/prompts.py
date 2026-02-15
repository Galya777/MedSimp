def build_simplification_prompt(discharge_text, literacy_level):
    return f"""
You are a medical communication assistant.

Rewrite the following hospital discharge instructions
for a patient with {literacy_level} health literacy.

Rules:
- Use short sentences.
- Use simple vocabulary.
- Avoid medical jargon.
- Use bullet points.
- Keep all critical safety information.

Discharge Instructions:
{discharge_text}
"""
