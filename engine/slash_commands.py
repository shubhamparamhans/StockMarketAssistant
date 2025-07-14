# trader_assist/engine/slash_commands.py

from engine.state_store import load_json, save_json
from engine.glossary import load_glossary, save_glossary, explain_term
from datetime import datetime
from engine.llm_provider import query_llm

def handle_slash_command(cmd):
    parts = cmd.split()
    command = parts[0]

    if command == "/add_position":
        return add_position(parts)
    elif command == "/note":
        return add_note(" ".join(parts[1:]))
    elif command == "/watchlist":
        return handle_watchlist(parts)
    elif command == "/explain":
        return explain_term(parts[1] if len(parts) > 1 else "")
    elif command == "/help":
        return HELP_TEXT
    elif command == "/refresh_glossary":
        
        handle_refresh_glossary()
    else:
        return f"❌ Unknown command: {command}"

def add_position(parts):
    if len(parts) < 5:
        return "Usage: /add_position TICKER QTY PRICE TAG"

    ticker, qty, price, tag = parts[1], int(parts[2]), float(parts[3]), parts[4]
    data = load_json("positions.json")
    data.append({
        "ticker": ticker,
        "qty": qty,
        "price": price,
        "tag": tag,
        "timestamp": datetime.now().isoformat()
    })
    save_json("positions.json", data)
    return f"✅ Added position: {ticker} x{qty} @ {price} ({tag})"

def add_note(note):
    data = load_json("notes.json")
    data.append({"note": note, "timestamp": datetime.now().isoformat()})
    save_json("notes.json", data)
    return "📝 Note added."

def handle_watchlist(parts):
    if len(parts) < 4 or parts[1] != "add":
        return "Usage: /watchlist add TICKER TARGET"

    ticker, target = parts[2], float(parts[3])
    data = load_json("watchlist.json")
    data.append({"ticker": ticker, "target": target})
    save_json("watchlist.json", data)
    return f"👁️ Added to watchlist: {ticker} → ₹{target}"

PREDEFINED_TERMS = [
    "ROCE", "ROE", "D/E", "P/E", "RSI", "MACD", "EPS", "FCF", "ADX",
    "VWAP", "CAGR", "Volume", "Beta", "Alpha", "NAV", "Intrinsic Value"
]

def handle_refresh_glossary():
    glossary = load_glossary()
    updated = False

    for term in PREDEFINED_TERMS:
        key = term.lower()
        if key in glossary:
            print(f"✅ {term}: already exists.")
            continue

        confirm = input(f"❓ Missing: '{term}'. Query LLM to explain and save? [y/N]: ").strip().lower()
        if confirm != 'y':
            continue

        prompt = f"Explain the financial term '{term}' in simple language for an Indian retail investor."
        definition = query_llm(prompt)
        glossary[key] = definition
        print(f"📘 {term} saved:\n{definition}\n")
        updated = True

    if updated:
        save_glossary(glossary)
        print("✅ Glossary updated and saved.")
    else:
        print("🔕 No changes made.")

HELP_TEXT = """
📘 Slash Commands:
/add_position TICKER QTY PRICE TAG
/note TEXT
/watchlist add TICKER TARGET
/explain TERM
/help
"""