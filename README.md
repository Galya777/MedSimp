# Adaptive Discharge AI

Adaptive Discharge AI is a human-centered AI application that uses the HAI-DEF model (MedGemma) to adapt clinical discharge instructions to the patient’s profile/health literacy. Goal: better understanding, higher adherence, and lower risk of readmission.

## Key features

- Automatic extraction of key sections from discharge text (Medications, Follow-up, Safety, Wound care, Emergency, Other).
- Rewriting instructions based on patient profile: `low_literacy`, `elderly`, `standard` (short sentences, no added information, no changes to numbers).
- Post-validation framework (numbers and critical terms must be preserved; readability before/after).
- Lightweight web interface (Streamlit) for quick demos.

## Technologies

- Python 3.10+
- PyTorch + Hugging Face Transformers
- MedGemma (HAI-DEF)
- BitsAndBytes for 4-bit quantization (GPU)
- Streamlit for UI
- textstat / readability-lxml for readability

## Modes of operation

1) Full (recommended, Colab/GPU): loads MedGemma in 4-bit (`bitsandbytes`) and performs section-wise regeneration. Suitable for demos and video recording.
2) Lite (local, 4 GB RAM): UI and steps for section extraction; LLM inference is limited/slow on CPU and not recommended. Use Full mode in Colab for real results.

## Installation (local)

```bash
git clone <repo_url>
cd MedSimp
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: set the model via environment variable (default is `google/medgemma-4b-it`):

```bash
export MODEL_ID=google/medgemma-4b-it
```

## Launch the UI (Streamlit)

```bash
streamlit run add/main.py
```

Enter the discharge text in the input, choose a profile, and click "Simplify." On a machine without a GPU, the model may be too heavy — use Full mode in Colab.

## Performance and memory notes

- On GPU: the model loads in 4-bit (NF4) via `bitsandbytes` automatically if CUDA is available.
- On CPU: loading is fp32 and intended only for short smoke tests. For 4 GB RAM we recommend Colab.

## Project structure

```
MedSimp/
├─ add/
│  ├─ main.py          # Streamlit UI
│  ├─ inference.py     # Model loading and section-wise generation
│  ├─ structure.py     # Deterministic section extraction
│  ├─ prompts.py       # Helper prompts (optional)
│  ├─ litteracy.py     # Profile modifiers and readability utilities
│  └─ test_prompt.py   # Example CLI test
├─ requirements.txt
└─ README.md
```

## Quick console test

```bash
python add/test_prompt.py
```

This extracts sections from a sample text, loads the model, and generates adapted instructions based on the selected profile.

## Reproducibility (for the competition)

- Public repo (this project).
- Full mode via Google Colab (T4/L4 GPU) with `bitsandbytes` and pinned versions from `requirements.txt`.
- Lite local mode to demonstrate the UI and pipeline without heavy inference.

## Colab Notebook (official repro path)

- Notebook: `notebooks/Adaptive_Discharge_AI_Colab.ipynb` (open in Google Colab; set Runtime → GPU).
- Alternative: use `add/colab_demo.py` directly in a Colab environment.

## Validation and metrics

- Synthetic set: `data/synthetic_validation.jsonl` (no PHI).
- Metrics script: `python add/metrics_eval.py` → writes `data/metrics_results.csv` and prints aggregates (FKGL/SMOG, length ratio, numbers preserved).

## Known limitations / FAQ

- "Can it run locally on 4 GB RAM?" — MedGemma inference on CPU is heavy and not recommended. Use Colab/GPU (Full mode). The local "Lite" mode is for UI and sectioning.
- "Why MedGemma?" — An open, domain-adapted model for medical text. More suitable for terminology and safety compared to general-purpose LLMs.
- "How do you ensure safety?" — Section-wise regeneration, strict prompts, and post-validation: numbers/units/critical terms are checked; on violation → strict retry → safe fallback.

## License and responsibility

This project is for demonstration purposes and does not replace medical advice. Do not include PHI/personal data. Content must be reviewed by a clinician before any real-world use.