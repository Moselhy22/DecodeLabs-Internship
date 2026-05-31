"""
DecodeLabs Internship - Project 1: Rule-Based AI Chatbot
Core chatbot engine with dictionary-based response matching.

Architecture: IPO Model
    INPUT    -> Sanitization & Normalization (utils.py)
    PROCESS  -> Intent Matching & State (this file)
    OUTPUT   -> Response Generation (responses.py)

The Heartbeat: Infinite while loop with kill command.
"""

from responses import RESPONSES, FALLBACK_RESPONSE, EXIT_COMMANDS
from utils import sanitize_input, is_empty_input, format_bot_response


def get_response(user_input: str) -> str:
    """
    Process sanitized user input and return appropriate response.

    Uses O(1) dictionary lookup instead of O(n) if-elif ladder.
    Falls back to default response for unknown intents.

    Args:
        user_input: Raw string from user

    Returns:
        Formatted bot response string
    """
    # Phase 1: Sanitize input
    cleaned = sanitize_input(user_input)

    # Handle empty input
    if is_empty_input(cleaned):
        return format_bot_response("I didn't catch that. Could you say something?")

    # Phase 2: O(1) Dictionary Lookup + Fallback
    # .get() is atomic: lookup + fallback in single operation
    reply = RESPONSES.get(cleaned, FALLBACK_RESPONSE)

    # Phase 3: Format and return
    return format_bot_response(reply)


def is_exit_command(user_input: str) -> bool:
    """
    Check if user wants to end the conversation.

    Args:
        user_input: Raw string from user

    Returns:
        True if exit command detected, False otherwise
    """
    cleaned = sanitize_input(user_input)
    return cleaned in EXIT_COMMANDS


def run_chatbot():
    """
    Main chatbot loop — The Heartbeat.

    Continuously accepts user input until exit command is given.
    Demonstrates the infinite cycle with kill command pattern.
    """
    # Welcome message
    print("=" * 50)
    print("🤖  DecodeBot - Rule-Based AI Chatbot")
    print("=" * 50)
    print("Type 'help' to see what I can do.")
    print("Type 'bye', 'exit', or 'quit' to end the chat.")
    print("-" * 50)
    print()

    # The Infinite Cycle (The Heartbeat)
    while True:
        # INPUT: Get raw user input
        try:
            raw_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n🤖 DecodeBot: Session interrupted. Goodbye!")
            break

        # KILL COMMAND: Check for exit
        if is_exit_command(raw_input):
            # Get the goodbye response before breaking
            goodbye = RESPONSES.get(sanitize_input(raw_input), "Goodbye!")
            print(format_bot_response(goodbye))
            print("=" * 50)
            break

        # PROCESS + OUTPUT: Get response and print
        response = get_response(raw_input)
        print(response)
        print()


if __name__ == "__main__":
    run_chatbot()
