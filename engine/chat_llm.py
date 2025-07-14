# trader_assist/engine/chat_llm.py

import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def handle_chat(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a trading assistant. Be concise, helpful, and explain any concept asked."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"