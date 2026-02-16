import re

SECTION_KEYWORDS = {
    "medications": [
        "medication", "medications", "pills", "tablet", "dose", "mg"
    ],
    "follow_up": [
        "follow", "appointment", "clinic", "schedule"
    ],
    "safety": [
        "drive", "machinery", "fall", "dizzy", "lightheaded"
    ],
    "wound_care": [
        "incision", "wound", "dressing", "bandage"
    ],
    "emergency": [
        "911", "emergency", "bleeding", "chest pain", "difficulty breathing"
    ]
}


def extract_sections(text: str) -> dict:
    """
    Deterministically extract discharge instruction sections.
    """

    sections = {
        "medications": [],
        "follow_up": [],
        "safety": [],
        "wound_care": [],
        "emergency": [],
        "other": []
    }

    lines = text.split("\n")

    for line in lines:
        clean_line = line.strip().lower()

        if not clean_line:
            continue

        matched = False

        for section, keywords in SECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in clean_line:
                    sections[section].append(line.strip())
                    matched = True
                    break
            if matched:
                break

        if not matched:
            sections["other"].append(line.strip())

    # convert lists to text blocks
    for key in sections:
        sections[key] = "\n".join(sections[key])

    return sections
