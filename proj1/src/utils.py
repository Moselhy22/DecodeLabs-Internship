"""
DecodeLabs Internship - Project 1: Rule-Based AI Chatbot
Utility functions for input sanitization and normalization.

Phase 1: INPUT & SANITIZATION
- Raw user input -> Cleaned, normalized string
- Handles case variations, whitespace, punctuation
- Prepares input for O(1) dictionary lookup in responses.py
"""

import re


def sanitize_input(raw_input: str) -> str:
    """
    Sanitize and normalize raw user input for dictionary matching.

    Pipeline:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Remove extra internal whitespace
        4. Remove common punctuation (optional, for flexibility)

    Args:
        raw_input: The raw string from user input()

    Returns:
        A clean, normalized string ready for dictionary key matching

    Examples:
        >>> sanitize_input("  HeLLo  ")
        'hello'
        >>> sanitize_input("  WHAT TIME IS IT???  ")
        'what time is it'
        >>> sanitize_input("   ")
        ''
    """
    if not raw_input:
        return ""

    # Step 1: Strip leading/trailing whitespace
    cleaned = raw_input.strip()

    # Step 2: Convert to lowercase for case-insensitive matching
    cleaned = cleaned.lower()

    # Step 3: Remove common punctuation (keeps alphanumeric and spaces)
    # This makes "hello!" match "hello" and "what's up?" match "whats up"
    cleaned = re.sub(r'[^\w\s]', '', cleaned)

    # Step 4: Collapse multiple spaces into one
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def is_empty_input(cleaned_input: str) -> bool:
    """
    Check if the sanitized input is empty.

    Args:
        cleaned_input: Already sanitized string

    Returns:
        True if empty, False otherwise
    """
    return len(cleaned_input) == 0


def format_bot_response(response_text: str) -> str:
    """
    Format the bot's response with a consistent prefix.

    Args:
        response_text: The raw response string

    Returns:
        Formatted response with bot prefix
    """
    return f"🤖 DecodeBot: {response_text}"
