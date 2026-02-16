def build_prompt(instruction_line: str, profile: str):
    base_instruction = """
You are a clinical discharge assistant.

Rewrite the following instruction clearly.
Do NOT add text, explanations, summaries, or code blocks.
Output exactly one rewritten sentence per bullet.
Do NOT change numbers, times, or quantities.
Do NOT add new information.
Do NOT add new instructions.
Do NOT invent details.
Keep meaning identical.
Keep it short.
Output ONLY one bullet point.
"""

    profile_modifiers = {
        "low_literacy": "Use very simple words. Short sentences.",
        "elderly": "Use calm tone. Emphasize safety.",
        "post_surgery": "Be precise. Emphasize recovery safety.",
        "chronic_condition": "Emphasize medication adherence."
    }

    full_prompt = f"""
{base_instruction}

Patient profile:
{profile_modifiers.get(profile, "")}

Instruction:
{instruction_line}

Rewrite:
"""

    return full_prompt
