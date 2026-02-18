import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Default HAI-DEF model (can be overridden via env var MODEL_ID)
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "google/medgemma-4b-it")


def load_model():
    """
    Load MedGemma model with best available configuration.
    - If CUDA is available: load in 4-bit with bitsandbytes for memory efficiency.
    - Else: fall back to CPU fp32 (may be slow and require more memory). For 4 GB RAM, prefer running via Colab.
    """

    model_id = DEFAULT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # CPU fallback; for constrained RAM this is only for smoke tests with tiny prompts
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        )

    model.eval()
    return tokenizer, model



def _extract_numbers_units(text: str):
    """Extract numbers (integers/decimals) and common units tokens for validation."""
    # Numbers like 2, 2.5, 0.25, etc.
    nums = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)
    # Common simple units/keywords to preserve (extend as needed)
    units = re.findall(r"\b(mg|ml|hrs?|hours?|days?|weeks?|bpm|mmhg)\b", text, flags=re.IGNORECASE)
    # Emergency keywords
    critical = re.findall(r"\b(911|emergency|chest pain|difficulty breathing)\b", text, flags=re.IGNORECASE)
    # Normalize dots/commas in numbers
    nums_norm = [n.replace(',', '.') for n in nums]
    return set(nums_norm), set(u.lower() for u in units), set(c.lower() for c in critical)


def _passes_safety(original: str, generated: str) -> bool:
    o_nums, o_units, o_crit = _extract_numbers_units(original)
    g_nums, g_units, g_crit = _extract_numbers_units(generated)
    return (o_nums == g_nums) and (o_units.issubset(g_units)) and (o_crit.issubset(g_crit))


def generate_response(tokenizer, model, sections, profile):

    profile_instruction = {
        "elderly": "Use very simple language. Short sentences. Focus on safety and fall prevention.",
        "low_literacy": "Use 6th grade reading level. Short sentences. Bullet points only.",
        "standard": "Rewrite clearly and professionally.",
        "post_surgery": "Use simple, calm language. Emphasize incision care, rest, and when to call a doctor.",
        "cardiac": "Use simple language. Emphasize chest pain, shortness of breath, and when to call 911."
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

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Cut prompt tokens -> keep only generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        cleaned = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Safety post-validation: ensure numbers/units/critical keywords are preserved
        if section_text.strip():
            if not _passes_safety(section_text, cleaned):
                # One conservative retry with stricter instruction
                strict_prompt = (
                    prompt
                    + "\n\nABSOLUTE CONSTRAINTS: Copy ALL original numbers and units exactly as in the section."
                    + " If you remove any critical words (e.g., 911, emergency, chest pain, difficulty breathing), add them back."
                )
                strict_inputs = tokenizer(strict_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    retry = model.generate(
                        **strict_inputs,
                        max_new_tokens=160,
                        do_sample=False,
                        temperature=0.0,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                retry_tokens = retry[0][strict_inputs["input_ids"].shape[-1]:]
                retry_clean = tokenizer.decode(retry_tokens, skip_special_tokens=True).strip()
                if _passes_safety(section_text, retry_clean):
                    cleaned = retry_clean
                else:
                    # Fallback: return original section text to avoid unsafe changes
                    cleaned = section_text.strip()

        final_output += f"\n\n## {section_name.capitalize()}\n"
        final_output += cleaned

    return final_output
