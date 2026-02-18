# Adaptive Discharge AI

Adaptive Discharge AI е human-centered AI приложение, което използва HAI-DEF модел (MedGemma) за адаптиране на клинични discharge инструкции според профила/здравната грамотност на пациента. Цел: по-добро разбиране, по-висока придържаност и по-нисък риск от повторна хоспитализация.

## Основни възможности

- Автоматично извличане на ключови секции от discharge текст (Medications, Follow-up, Safety, Wound care, Emergency, Other).
- Преписване на инструкциите според пациентски профил: `low_literacy`, `elderly`, `standard` (кратки изречения, без добавена информация, без промяна на числа).
- Пост-валидационна рамка (числата и критични думи трябва да се запазят; четивност преди/след).
- Лек уеб интерфейс (Streamlit) за бърза демонстрация.

## Технологии

- Python 3.10+
- PyTorch + Hugging Face Transformers
- MedGemma (HAI-DEF)
- BitsAndBytes за 4-bit quant (GPU)
- Streamlit за UI
- textstat / readability-lxml за четивност

## Режими на работа

1) Full (препоръчително, Colab/GPU): зарежда MedGemma в 4-bit (`bitsandbytes`) и изпълнява секционно прегенериране. Подходящ за демонстрация и запис на видео.
2) Lite (локално, 4 GB RAM): UI и стъпки по извличане на секции; LLM inference е ограничен/бавен на CPU и не е препоръчителен. Използвайте Full в Colab за реални резултати.

## Инсталация (локално)

```bash
git clone <repo_url>
cd MedSimp
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

По желание: задайте модел чрез променлива на средата (по подразбиране е `google/medgemma-4b-it`):

```bash
export MODEL_ID=google/medgemma-4b-it
```

## Стартиране на UI (Streamlit)

```bash
streamlit run add/main.py
```

Въведете discharge текст в полето, изберете профил и натиснете "Simplify". На машина без GPU моделът може да е твърде тежък – използвайте Full режим в Colab.

## Бележки за производителност и памет

- На GPU: моделът се зарежда в 4-bit (NF4) чрез `bitsandbytes` автоматично, ако има налична CUDA.
- На CPU: зареждането е fp32 и е предназначено само за кратки smoke тестове. За 4 GB RAM препоръчваме Colab.

## Структура на проекта

```
MedSimp/
├─ add/
│  ├─ main.py          # Streamlit UI
│  ├─ inference.py     # Зареждане на модела и генериране по секции
│  ├─ structure.py     # Детерминистично извличане на секции
│  ├─ prompts.py       # Помощни промптове (по избор)
│  ├─ litteracy.py     # Профилни модификатори и четивност (утилити)
│  └─ test_prompt.py   # Примерен CLI тест
├─ requirements.txt
└─ README.md
```

## Бърз тест от конзола

```bash
python add/test_prompt.py
```

Това извлича секции от примерен текст, зарежда модела и генерира адаптирани инструкции според профил.

## Репродуцируемост (за състезанието)

- Публично репо (този проект).
- Full режим през Google Colab (T4/Л4 GPU) с `bitsandbytes` и фиксирани версии от `requirements.txt`.
- Lite локален режим за демонстрация на UI и pipeline без тежко inference.

## Colab Notebook (официален репро път)

- Notebook: `notebooks/Adaptive_Discharge_AI_Colab.ipynb` (отворете в Google Colab; задайте Runtime → GPU).
- Алтернатива: използвайте `add/colab_demo.py` директно в Colab среда.

## Валидиране и метрики

- Синтетичен сет: `data/synthetic_validation.jsonl` (без PHI).
- Скрипт за метрики: `python add/metrics_eval.py` → записва `data/metrics_results.csv` и принтира агрегати (FKGL/SMOG, length ratio, numbers preserved).

## Known limitations / FAQ

- „Може ли локално на 4 GB RAM?“ — Инференсът на MedGemma на CPU е тежък и не е препоръчителен. Използвайте Colab/GPU (Full режим). Локалният „Lite“ режим е за UI и секциониране.
- „Защо MedGemma?“ — Отворен, домейн‑адаптиран модел за медицински текст. По‑подходящ за терминология и безопасност спрямо общи LLM‑и.
- „Как гарантирате безопасност?“ — Секционно прегенериране, строги промптове и пост‑валидация: числа/единици/критични думи се съпоставят; при нарушение → строг retry → безопасен fallback.

## Лиценз и отговорност

Проектът е демонстрационен и не замества медицински съвет. Не включвайте PHI/лични данни. Проверката на съдържанието от клиницист е задължителна преди реална употреба.