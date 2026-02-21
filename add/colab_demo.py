"""
Colab Full Demo
----------------
Инструкции (стартирайте в Google Colab):

1) Runtime → Change runtime type → GPU
2) Първа клетка (инсталация):

!pip -q install torch==2.3.1 transformers==4.48.0 accelerate==0.34.2 bitsandbytes==0.43.1 textstat==0.7.4 readability-lxml==0.8.1 sentencepiece==0.2.0 safetensors==0.4.3

3) Втора клетка: качете/клонирайте репото или копирайте този файл и нужните модули.
4) Трета клетка: изпълнете примера по-долу.

Забележка: Ако искате друг модел, задайте MODEL_ID преди изпълнение.
"""

import os
from add.inference import load_model, generate_response
from add.structure import extract_sections

def demo_run():
    os.environ.setdefault("MODEL_ID", "google/medgemma-4b-it")
    tokenizer, model = load_model()

    discharge_text = """
    - Take 10 mg lisinopril every morning.
    - Avoid driving for 24 hours after sedation.
    - Call 911 if you have chest pain or difficulty breathing.
    - Follow up with cardiology clinic in 7 days.
    """.strip()

    sections = extract_sections(discharge_text)
    out = generate_response(tokenizer, model, sections, profile="low_literacy")
    print("\n=== OUTPUT (low_literacy) ===\n")
    print(out)


if __name__ == "__main__":
    demo_run()
