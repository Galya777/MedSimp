import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from prompts import build_prompt

MODEL_ID = "google/medgemma-4b-it"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    model.eval()
    return tokenizer, model



def generate_response(tokenizer, model, sections, profile):

    profile_instruction = {
        "elderly": "Use very simple language. Short sentences. Focus on safety and fall prevention.",
        "low_literacy": "Use 6th grade reading level. Short sentences. Bullet points only.",
        "standard": "Rewrite clearly and professionally."
    }

    final_output = ""

    for section_name, section_text in sections.items():

        prompt = f"""
You are a medical assistant.

Rewrite the following section for a {profile} patient.

{profile_instruction.get(profile, "")}

STRICT RULES:
- Do NOT add new information.
- Do NOT remove information.
- Do NOT change numbers.
- Output ONLY the rewritten text.
- No code.
- No explanations.

Section:
{section_text}
"""

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Cut prompt tokens -> keep only generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        cleaned = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        final_output += f"\n\n## {section_name.capitalize()}\n"
        final_output += cleaned

    return final_output
