from typing import Dict

try:
    import textstat
except Exception:  # optional dependency for Lite mode
    textstat = None


PROFILE_MODIFIERS: Dict[str, str] = {
    "low_literacy": "Use very simple words. Short sentences. Bullet points only.",
    "elderly": "Use calm tone. Short sentences. Emphasize safety and fall prevention.",
    "standard": "Rewrite clearly and professionally.",
}


def get_profile_modifier(profile: str) -> str:
    return PROFILE_MODIFIERS.get(profile, PROFILE_MODIFIERS["standard"])


def readability_scores(text: str) -> Dict[str, float]:
    """Return common readability metrics if textstat is available."""
    if not textstat:
        return {"fkgl": -1.0, "smog": -1.0}
    try:
        fkgl = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        fkgl = -1.0
    try:
        smog = float(textstat.smog_index(text))
    except Exception:
        smog = -1.0
    return {"fkgl": fkgl, "smog": smog}
