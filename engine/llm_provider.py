# engine/llm_provider.py

import requests
import openai
from config import LLM_PROVIDER, OPENAI_API_KEY, API_URL

def query_llm(prompt: str, history: list = None) -> str:
    history = history or []

    if LLM_PROVIDER == "openai":
        openai.api_key = OPENAI_API_KEY
        messages = [{"role": "system", "content": "You are a helpful trading assistant."}]
        messages += [{"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ OpenAI error: {str(e)}"

    elif LLM_PROVIDER == "deepseek":
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={"messages": [{"role": "user", "content": prompt}], "model": "deepseek-chat"}
            )
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"❌ DeepSeek error: {str(e)}"

    elif LLM_PROVIDER == "ollama":
        try:
            response = requests.post(
                API_URL,
                json={"prompt": prompt, "model": "llama3"}
            )
            return response.json()["response"].strip()
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    else:
        return f"❌ Unknown LLM provider: {LLM_PROVIDER}"