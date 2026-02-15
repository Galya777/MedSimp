from inference import load_model, generate_response

# Зареждаме модела и токенизатора
tokenizer, model = load_model()

# Примерен prompt за тест
prompt = """You are a medical assistant. Rewrite the following discharge instructions for a patient with 6th grade literacy level. Use short sentences, simple words, and bullet points.

Discharge Instructions:
- Take 2 pills daily
- Avoid driving for 24 hours
- Follow up in 7 days"""

# Генерираме опростената версия на текста
response = generate_response(tokenizer, model, prompt)

# Показваме резултата
print("=== Simplified Instructions ===")
print(response)
