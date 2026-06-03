# 🏗️ Technical Architecture: Rule-Based AI Chatbot

## 1. Design Philosophy

This chatbot follows the **IPO Model** (Input-Process-Output) with a strict separation of concerns:

```text
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   INPUT     │ --> │    PROCESS      │ --> │    OUTPUT       │
│  Sanitize   │     │   Match Intent  │     │   Generate      │
│  Normalize  │     │   (O(1) Dict)   │     │   Response      │
└─────────────┘     └─────────────────┘     └─────────────────┘
```

## 2. Module Breakdown

### 2.1 `responses.py` — The Knowledge Base

**Data Structure:** Python Dictionary (`dict`)

### Why a Dictionary?

* **O(1) average lookup time**
* Direct access through hashing
* Clean and scalable design

```python
RESPONSES = {
    "hello": "Hello there! 👋",
    "hi": "Hi! Nice to meet you."
}
```

**Key Design Decision:** All keys are pre-sanitized before lookup.

---

### 2.2 `utils.py` — The Sanitization Pipeline

Pipeline:

1. `.strip()`
2. `.lower()`
3. Remove punctuation
4. Collapse extra spaces

Example:

```text
Raw Input:    "  HeLLo!!!  "
After Strip:  "HeLLo!!!"
After Lower:  "hello!!!"
After Clean:  "hello"
Final Key:    "hello"
```

---

### 2.3 `chatbot.py` — The Engine

Core loop:

```python
while True:
    raw_input = input("You: ")

    if is_exit_command(raw_input):
        print(goodbye_response)
        break

    response = get_response(raw_input)
    print(response)
```

### Why Use `while True`?

* Continuous conversation
* Simple architecture
* Easy exit handling

---

### 2.4 `main.py` — Entry Point

```python
if __name__ == "__main__":
    main()
```

This prevents automatic execution when imported as a module.

---

## 3. Algorithmic Complexity Analysis

| Operation          | Complexity |
| ------------------ | ---------- |
| Dictionary Lookup  | O(1)       |
| Sanitization       | O(m)       |
| Exit Command Check | O(1)       |

Where:

* `m` = length of user input

Total complexity per interaction:

```text
O(1) + O(m) = O(m)
```

---

## 4. Testing Strategy

| Test Layer     | Purpose                  | Count |
| -------------- | ------------------------ | ----- |
| Unit Tests     | Individual functions     | 20    |
| Edge Cases     | Empty input, punctuation | 6     |
| Data Integrity | Structure validation     | 3     |

Testing framework:

```bash
pytest
```

---

## 5. Extension Points

Adding a new intent:

1. Add key-value pair in `responses.py`
2. Add test in `test_chatbot.py`
3. Run pytest

No modifications are required in the chatbot engine.

---

## 6. The Bridge to Machine Learning

Current architecture:

```text
KEY ── exact match ──> VALUE
```

Future AI systems:

```text
VECTOR ── similarity search ──> MEANING
```

This project establishes the deterministic foundation before moving into probabilistic AI systems.

---

*DecodeLabs Internship — Project 1 Architecture Documentation*
