# trader_assist/config.py

import os
from pathlib import Path

CONFIG_FILE = Path(".openai_config")

def get_openai_api_key():
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    if CONFIG_FILE.exists():
        return CONFIG_FILE.read_text().strip()

    try:
        user_input = input("🔐 Enter your OpenAI API key: ").strip()
        if not user_input:
            print("❌ No API key provided. Exiting.")
            exit(1)

        CONFIG_FILE.write_text(user_input)
        print("✅ API key saved to .openai_config for future use.")
        return user_input
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled.")
        exit(1)

OPENAI_API_KEY = get_openai_api_key()