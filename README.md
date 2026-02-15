# Adaptive Discharge AI

**Adaptive Discharge AI** е human-centered AI проект, който използва MedGemma (HAI-DEF) за опростяване на медицински discharge инструкции според здравната грамотност на пациента. Целта е да се подобри разбирането на пациентите и да се намали риска от повторно хоспитализиране.

## Основни функции

- Анализира discharge summary и извлича ключови инструкции.
- Адаптира текста според зададено ниво на literacy (например 6th grade, 10th grade, медицински персонал).
- Генерира кратки, ясни и безопасни инструкции с bullet points.
- Подготвя output, готов за визуализация в уеб UI чрез Streamlit.

## Технологии и инструменти

- Python 3.10+
- PyTorch & Transformers
- MedGemma (HAI-DEF)
- BitsAndBytes за 4-bit/8-bit моделна оптимизация
- Streamlit за бърз и интерактивен UI
- Pandas / scikit-learn / readability-lxml за обработка на текст

## Инсталация и setup

1. Клонирайте проекта:

```bash
git clone <repo_url>
cd MedSimp


TO BE DONE!!!