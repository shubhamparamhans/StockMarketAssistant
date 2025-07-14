# engine/glossary.py

import json
import os
from engine.llm_provider import query_llm

GLOSSARY_FILE = "data/glossary.json"

# Load glossary from file
def load_glossary():
    if not os.path.exists(GLOSSARY_FILE):
        return {}
    with open(GLOSSARY_FILE, "r") as f:
        return json.load(f)

# Save glossary back to file
def save_glossary(glossary):
    with open(GLOSSARY_FILE, "w") as f:
        json.dump(glossary, f, indent=2)

# Explain term with optional LLM fallback
def explain_term(term: str, allow_llm_fallback: bool = True) -> str:
    glossary = load_glossary()
    key = term.strip().lower()

    if key in glossary:
        return f"📘 {term.upper()}:\n{glossary[key]}"

    if not allow_llm_fallback:
        return f"🤔 No explanation found for '{term}'."

    confirmation = input(f"❓ Term '{term}' not found. Ask LLM for explanation? [y/N]: ").strip().lower()
    if confirmation != 'y':
        return "🔕 Skipped querying LLM."

    prompt = f"Explain the financial term '{term}' in simple, concise language suitable for an Indian retail investor."
    definition = query_llm(prompt)

    glossary[key] = definition
    save_glossary(glossary)
    return f"📘 {term.upper()} (via LLM):\n{definition}"