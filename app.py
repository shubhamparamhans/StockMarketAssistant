# trader_assist/app.py

from engine.slash_commands import handle_slash_command
from engine.chat_llm import handle_chat
import readline

def main():
    print("👋 Welcome to Trader Assist CLI")
    print("Type `/help` for commands or ask anything...\n")

    while True:
        try:
            user_input = input("🧠 > ").strip()

            if not user_input:
                continue
            elif user_input.startswith("/"):
                response = handle_slash_command(user_input)
            else:
                response = handle_chat(user_input)

            print(response + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting Trader Assist.")
            break

if __name__ == "__main__":
    main()
