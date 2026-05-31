"""
DecodeLabs Internship - Project 1: Rule-Based AI Chatbot
Entry point to run the chatbot.

Usage:
    python main.py

This script imports and runs the chatbot engine from src/chatbot.
"""

import sys
from pathlib import Path

# Add src/ to Python path so we can import our modules
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from chatbot import run_chatbot


def main():
    """
    Main entry point for the DecodeBot chatbot application.
    """
    try:
        run_chatbot()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
