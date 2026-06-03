# 🤖 Project 1: Rule-Based AI Chatbot

> **DecodeLabs Internship — Batch 2026**
>
> *"Before you can manage the chaos of a probability engine, you must master the precision of a logic engine."*

---

## 📋 Project Overview

A deterministic, rule-based AI chatbot built entirely with Python dictionaries and control flow logic. This project demonstrates the foundational architecture of intelligent interfaces — **no machine learning, no neural networks, just pure programmatic decision-making.**

### Key Concepts Demonstrated

* **O(1) Dictionary Lookup** vs. O(n) if-elif anti-patterns
* **Input Sanitization & Normalization** (case, whitespace, punctuation)
* **The Infinite Loop with Kill Command** (continuous interaction)
* **Fallback Strategy** for unknown inputs
* **Deterministic "White Box" AI** — 100% traceable, zero hallucination risk

---

## 🏗️ Architecture (IPO Model)

```text
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   INPUT     │ --> │    PROCESS      │ --> │    OUTPUT       │
│  (Raw Feed) │     │ (Logic Skeleton)│     │ (Feedback Loop) │
└─────────────┘     └─────────────────┘     └─────────────────┘
       │                   │                       │
       ▼                   ▼                       ▼
  Sanitization        Intent Matching          Response
  & Normalization     & State (O(1) dict)      Generation
```

| Component               | File                    | Purpose                               |
| ----------------------- | ----------------------- | ------------------------------------- |
| **Knowledge Base**      | `src/responses.py`      | Dictionary of 23+ intents → responses |
| **Sanitization Engine** | `src/utils.py`          | Cleans raw input for matching         |
| **Chatbot Engine**      | `src/chatbot.py`        | O(1) lookup, loop, exit handling      |
| **Entry Point**         | `main.py`               | Runs the application                  |
| **Tests**               | `tests/test_chatbot.py` | 20 unit tests with pytest             |

---

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* pytest (for testing)

### Installation

```bash
cd proj1
pip install -r requirements.txt
```

### Run the Chatbot

```bash
python3 main.py
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

---

## 💬 Supported Intents (23+)

| Category        | Commands                             | Example Response    |
| --------------- | ------------------------------------ | ------------------- |
| **Greetings**   | `hello`, `hi`, `hey`, `good morning` | "Hello there! 👋"   |
| **Identity**    | `who are you`, `what is your name`   | Self-introduction   |
| **Time**        | `what time is it`                    | Humorous time reply |
| **Help**        | `help`, `what should i do`           | Command list        |
| **Personality** | `how are you`, `tell me a joke`      | Jokes & status      |
| **Exit**        | `bye`, `exit`, `quit`, `goodbye`     | Graceful shutdown   |

---

## 🧪 Test Coverage

| Test Category           | Count  | Status           |
| ----------------------- | ------ | ---------------- |
| Intent Matching         | 8      | ✅ All Pass       |
| Sanitization Edge Cases | 4      | ✅ All Pass       |
| Fallback & Empty Input  | 2      | ✅ All Pass       |
| Exit Commands           | 3      | ✅ All Pass       |
| Data Integrity          | 3      | ✅ All Pass       |
| **Total**               | **20** | **✅ 20/20 Pass** |

---

## 📚 Why Dictionaries Beat If-Elif Ladders

| Metric              | If-Elif Ladder              | Dictionary (This Project) |
| ------------------- | --------------------------- | ------------------------- |
| **Time Complexity** | O(n) — grows with rules     | O(1) — constant time      |
| **Maintainability** | High technical debt         | Easy to extend            |
| **Readability**     | Deep nesting                | Clean key-value pairs     |
| **Scalability**     | Fragile, cascading failures | Robust, instant lookup    |

> *"The rule-based generative program is a white box. The interpreter can always provide straightforward explanations."*

---

## 🛡️ AI Guardrails Context

This project represents the **deterministic control layer** that sits in front of modern LLMs (like NVIDIA NeMo Guardrails or Llama Guard). It provides:

* **Traceability**: Input → Logic → Output. No mystery.
* **Safety**: Zero hallucination risk. 100% hard-coded.
* **Compliance**: Essential for Finance & Healthcare applications.

---

## 📁 File Structure

```text
proj1/
├── src/
│   ├── __init__.py
│   ├── responses.py
│   ├── utils.py
│   └── chatbot.py
├── tests/
│   ├── __init__.py
│   └── test_chatbot.py
├── docs/
│   └── ARCHITECTURE.md
├── main.py
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**DecodeLabs Internship — Batch 2026**

Built with deterministic logic, tested with precision, documented with care.

---

> *"An LLM without rules is a hallucination engine. Today, we build the skeleton that holds the intelligence."*
