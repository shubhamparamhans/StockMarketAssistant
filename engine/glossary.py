# trader_assist/engine/glossary.py

GLOSSARY = {
    "roce": "ROCE (Return on Capital Employed) measures how efficiently a company uses its capital to generate profit.",
    "rsi": "RSI (Relative Strength Index) indicates overbought (>70) or oversold (<30) market conditions.",
    "d/e": "D/E Ratio (Debt to Equity) compares a company’s total debt to shareholder equity.",
    "pe": "P/E Ratio (Price to Earnings) reflects how much investors are willing to pay per rupee of earnings.",
    "pb": "P/B Ratio (Price to Book) compares the market price to the book value of the stock.",
}

def explain_term(term):
    key = term.lower().strip()
    return GLOSSARY.get(key, f"🤔 No explanation found for '{term}'.")