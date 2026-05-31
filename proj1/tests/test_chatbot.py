"""
DecodeLabs Internship - Project 1: Rule-Based AI Chatbot
Unit tests for the rule-based chatbot.

Tests cover:
- Intent matching (greetings, identity, time, help, personality)
- Case insensitivity
- Punctuation stripping
- Fallback responses
- Empty input handling
- Exit command detection
"""

import sys
from pathlib import Path

# Add src/ to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from chatbot import get_response, is_exit_command
from responses import RESPONSES, FALLBACK_RESPONSE, EXIT_COMMANDS


# =============================================================================
# TEST: Intent Matching
# =============================================================================

def test_greeting_hello():
    """Test basic greeting intent."""
    result = get_response("hello")
    assert "Hello there" in result
    assert "🤖 DecodeBot:" in result


def test_greeting_hi():
    """Test 'hi' greeting variant."""
    result = get_response("hi")
    assert "Nice to meet you" in result


def test_greeting_hey():
    """Test 'hey' greeting variant."""
    result = get_response("hey")
    assert "Ready to help" in result


def test_identity_who_are_you():
    """Test identity intent."""
    result = get_response("who are you")
    assert "DecodeBot" in result
    assert "rule-based" in result


def test_time_intent():
    """Test time-related intent."""
    result = get_response("what time is it")
    assert "clock" in result or "time" in result


def test_help_intent():
    """Test help intent."""
    result = get_response("help")
    assert "hello" in result
    assert "bye" in result


def test_joke_intent():
    """Test joke intent."""
    result = get_response("tell me a joke")
    assert "Python" in result or "programmer" in result


def test_how_are_you():
    """Test personality intent."""
    result = get_response("how are you")
    assert "100%" in result or "functioning" in result


# =============================================================================
# TEST: Sanitization Edge Cases
# =============================================================================

def test_case_insensitive():
    """Test that uppercase input matches lowercase keys."""
    result_upper = get_response("HELLO")
    result_lower = get_response("hello")
    assert result_upper == result_lower


def test_leading_trailing_whitespace():
    """Test that extra whitespace is stripped."""
    result = get_response("   hello   ")
    assert "Hello there" in result


def test_punctuation_stripping():
    """Test that punctuation is removed before matching."""
    result = get_response("hello!!!")
    assert "Hello there" in result


def test_mixed_case_and_punctuation():
    """Test combined sanitization: case + punctuation + whitespace."""
    result = get_response("  HeLLo!!!  ")
    assert "Hello there" in result


# =============================================================================
# TEST: Fallback & Edge Cases
# =============================================================================

def test_unknown_input_fallback():
    """Test that unknown input returns fallback response."""
    result = get_response("this_is_not_a_known_intent_12345")
    assert "sorry" in result or "understand" in result


def test_empty_input():
    """Test that empty/whitespace-only input is handled."""
    result = get_response("")
    assert "catch" in result or "say something" in result

    result_spaces = get_response("     ")
    assert "catch" in result_spaces or "say something" in result_spaces


# =============================================================================
# TEST: Exit Commands
# =============================================================================

def test_exit_commands():
    """Test all exit commands are recognized."""
    for cmd in EXIT_COMMANDS:
        assert is_exit_command(cmd) is True, f"Failed for: {cmd}"


def test_non_exit_commands():
    """Test that non-exit commands are not flagged."""
    assert is_exit_command("hello") is False
    assert is_exit_command("help") is False
    assert is_exit_command("who are you") is False


def test_exit_with_punctuation():
    """Test exit detection with punctuation (sanitization applied)."""
    assert is_exit_command("bye!") is True
    assert is_exit_command("EXIT!!!") is True
    assert is_exit_command("  quit  ") is True


# =============================================================================
# TEST: Response Dictionary Integrity
# =============================================================================

def test_responses_not_empty():
    """Verify the knowledge base has entries."""
    assert len(RESPONSES) >= 5  # Project requirement: 5+ intents


def test_all_responses_are_strings():
    """Verify all response values are strings."""
    for key, value in RESPONSES.items():
        assert isinstance(key, str)
        assert isinstance(value, str)
        assert len(value) > 0


def test_fallback_is_string():
    """Verify fallback response is a valid string."""
    assert isinstance(FALLBACK_RESPONSE, str)
    assert len(FALLBACK_RESPONSE) > 0
