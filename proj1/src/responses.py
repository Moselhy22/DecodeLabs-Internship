"""
DecodeLabs Internship - Project 1: Rule-Based AI Chatbot
Knowledge base: dictionary of intents and responses.

Architecture:
- Each KEY is a sanitized user input (lowercase, stripped)
- Each VALUE is the bot's response string
- Uses Python dict for O(1) constant-time lookup
- Avoids slow O(n) if-elif ladder anti-pattern

Project Requirement: 5+ intents minimum
"""

# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

RESPONSES = {
    # --- Greeting Intents ---
    "hello": "Hello there! 👋 Welcome! How can I assist you today?",
    "hi": "Hi! Nice to meet you. What can I do for you?",
    "hey": "Hey! Ready to help. What's on your mind?",
    "good morning": "Good morning! ☀️ Hope you have a productive day!",
    "good afternoon": "Good afternoon! How's your day going?",
    "good evening": "Good evening! 🌙 What can I help you with?",

    # --- Identity Intents ---
    "who are you": "I'm DecodeBot 🤖, a rule-based AI assistant built by DecodeLabs. I operate on deterministic logic — no neural networks here, just pure if-else precision!",
    "what is your name": "You can call me DecodeBot. I'm a deterministic rule-based chatbot.",
    "what can you do": "I can respond to predefined intents using dictionary-based O(1) lookup. Try asking me about greetings, time, help, or just chat!",

    # --- Time Intents ---
    "what time is it": "I don't have a real-time clock, but your system does! Try running 'date' in your terminal. ⏰",
    "what is the time": "Time flies when you're coding! Check your system clock for the exact time.",
    "tell me the time": "I wish I could, but I'm a static rule-based bot. Your OS knows better!",

    # --- Help Intents ---
    "help": "Sure! Here are some things you can ask me:\n  • hello / hi / hey\n  • who are you\n  • what time is it\n  • tell me a joke\n  • how are you\n  • bye / exit / quit",
    "what should i do": "Start by saying 'hello' or ask me 'who are you'. Type 'help' anytime for a full list!",

    # --- Personality Intents ---
    "how are you": "I'm functioning at 100% efficiency! All logic gates are green. 💚 How about you?",
    "how are you doing": "Running smoothly on deterministic logic. No hallucinations here! 🤖",
    "tell me a joke": "Why did the Python programmer break up with Java? Because she didn't get his 'class' references! 🐍",
    "are you a robot": "Yes! I'm a rule-based AI — a 'white box' system where every decision is traceable and explainable.",

    # --- Exit Intents ---
    "bye": "Goodbye! 👋 It was nice chatting with you. Come back anytime!",
    "goodbye": "See you later! Keep building amazing things! 🚀",
    "see you": "See you! Don't forget to commit your code! 😄",
    "exit": "Exiting... Have a great day!",
    "quit": "Quitting session. Stay curious, keep coding!",
}

# =============================================================================
# FALLBACK RESPONSE
# =============================================================================

FALLBACK_RESPONSE = (
    "I'm sorry, I didn't understand that. 🤔\n"
    "I'm a rule-based bot with predefined intents. "
    "Try typing 'help' to see what I can do!"
)

# =============================================================================
# EXIT COMMANDS
# =============================================================================

EXIT_COMMANDS = {"bye", "goodbye", "see you", "exit", "quit"}
