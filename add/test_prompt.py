from inference import load_model, generate_response
from structure import extract_sections

discharge_text = """
- Take 2 pills daily
- Avoid driving for 24 hours
- Follow up in 7 days
"""

profile = "elderly"

# 1️⃣ Extract sections
sections = extract_sections(discharge_text)

print("\n=== EXTRACTED SECTIONS ===")
for k, v in sections.items():
    print(f"\n--- {k.upper()} ---")
    print(v)

# 2️⃣ Load model
tokenizer, model = load_model()

# 3️⃣ Generate adaptive response
response = generate_response(tokenizer, model, sections, profile)

print("\n=== Adaptive Discharge Instructions ===")
print(response)
