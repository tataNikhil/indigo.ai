# Indigo.ai

Hands-on AI engineering project, built week by week: LLM APIs, structured outputs,
validation, response generation and conversation memory.


## Week 1 — Enterprise Structured Ingestion Firewall

Raw support or financial text goes in; a strictly typed JSON object comes out.
If the model's reply does not match the schema, Instructor re-asks the model; if it
still fails, the request is rejected with `502` and nothing reaches downstream systems.

