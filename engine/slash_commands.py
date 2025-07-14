# trader_assist/engine/slash_commands.py

from engine.state_store import load_json, save_json
from engine.glossary import explain_term
from datetime import datetime

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

HELP_TEXT = """
📘 Slash Commands:
/add_position TICKER QTY PRICE TAG
/note TEXT
/watchlist add TICKER TARGET
/explain TERM
/help
"""