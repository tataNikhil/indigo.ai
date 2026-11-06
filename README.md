# Indigo.ai

Hands-on AI engineering project, built week by week: LLM APIs, structured outputs,
validation, response generation and conversation memory.

| Week | Focus | Status |
|---|---|---|
| 1 | LLM APIs, structured outputs, Pydantic validation, Enterprise Structured Ingestion Firewall | ✅ |
| 2 | Intent-based AI response generator + React Markdown UI | ⏳ |
| 3 | Conversation memory and chat sessions | ⏳ |

## Repository layout

```text
Indigo.ai/
├── practice/week1/        # pre-session mini tasks
│   ├── task1_groq_check.py           # Task 1: verify Groq API key + model
│   ├── user_profile.py               # Pydantic UserProfile (name, age, email, PAN)
│   ├── task2_pydantic_validation.py  # Task 2: valid/invalid data + ValidationError
│   ├── task3_instructor_profile.py   # Task 3: Instructor response_model extraction
│   ├── regex_vs_schema.py            # why regex parsing fails on LLM output
│   └── token_budget.py               # tiktoken counting + context budgeting
└── backend/               # Enterprise Structured Ingestion Firewall (FastAPI)
    ├── app/
    │   ├── main.py                    # routes + error mapping
    │   ├── config.py                  # env settings
    │   ├── schemas/ingestion.py       # request / LLM / response models
    │   ├── services/llm_service.py    # Groq + Instructor call
    │   ├── services/firewall.py       # pipeline: budget -> LLM -> guard rules
    │   ├── services/injection_guard.py# deterministic injection + PII checks
    │   └── utils/token_counter.py     # tiktoken estimate
    └── tests/test_firewall.py         # runs with a fake LLM, no key needed
```

## Week 1 — Enterprise Structured Ingestion Firewall

Raw support or financial text goes in; a strictly typed JSON object comes out.
If the model's reply does not match the schema, Instructor re-asks the model; if it
still fails, the request is rejected with `502` and nothing reaches downstream systems.

```text
POST /api/v1/ingest
   │
   ├─ IngestionRequest validation (non-empty, ≤8000 chars, known channel, no extra fields)
   ├─ token budget check with tiktoken  ──► 413 if over MAX_INPUT_TOKENS
   ├─ Groq LLM via Instructor (response_model=LLMClassification, retries on invalid JSON)
   ├─ deterministic guard rules: prompt-injection + PII flags the LLM can't override
   └─ IngestionResult  (intent, risk profile, priority parameters, token metadata)
```

Example:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"message": "Rs 4,999 was debited twice from my account. Refund urgently!", "channel": "chat"}'
```

```json
{
  "intent": "payment_issue",
  "confidence": 0.94,
  "risk": {"level": "high", "flags": ["financial_loss"], "reason": "Customer lost money to a duplicate debit."},
  "priority": {"level": "high", "requires_human": true, "sla_hours": 4},
  "summary": "Customer reports a duplicate debit of Rs 4,999 and requests a refund.",
  "token_metadata": {"estimated_input_tokens": 19, "prompt_tokens": 612, "completion_tokens": 88,
                     "model": "openai/gpt-oss-120b", "latency_ms": 640}
}
```

### Validation rules worth knowing

- `intent`, `risk.level`, `risk.flags`, `priority.level` are `Literal` types: any other value is rejected.
- `confidence` must be between 0 and 1.
- Cross-field rule: `fraud_suspected` forces risk to `high` or `critical`.
- `sla_hours` is a business rule set in code from the priority, never by the model.
- If the message tries to steer the classifier ("ignore previous instructions…"),
  `prompt_injection` is added, risk is raised to at least `medium` and a human is required,
  even when the model was fooled.
- `POST /api/v1/ingest/batch` accepts a JSON array (Pydantic `RootModel`) of 1–10 messages.

## Setup

Requires Python 3.10+ and a free Groq API key from https://console.groq.com/keys.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
# put your key in .env -> GROQ_API_KEY=...

pytest -q                          # tests use a fake LLM, no key needed
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

Practice tasks (run from `practice/week1`, with `GROQ_API_KEY` in a `.env` there or exported):

```bash
python task1_groq_check.py
python task2_pydantic_validation.py
python task3_instructor_profile.py "I'm Priya Sharma, 31, priya@mail.com, PAN bnzps4567q"
python regex_vs_schema.py
python token_budget.py
```

### Free-tier notes

Groq's free tier is rate limited per minute and per day. A `429` from Groq surfaces as
`502` from this API; wait and retry. `tiktoken` counts are estimates for Groq models;
`prompt_tokens` / `completion_tokens` in the response are the provider's own numbers.
