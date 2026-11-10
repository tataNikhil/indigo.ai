# Indigo.ai

Hands-on AI engineering project, built week by week: LLM APIs, structured outputs,
validation, response generation and conversation memory.

## Week 1 — Enterprise Structured Ingestion Firewall

Raw support or financial text goes in; a strictly typed JSON object comes out, with the
intent, risk profile, priority and token metadata. If the model's reply does not match the
schema, Instructor re-asks the model; if it still fails, the request is rejected and
nothing reaches downstream systems.

## Week 2 — Intent-Based AI Response Generator

The firewall's analysis is used as context for a second LLM call that writes the actual
reply in Markdown, using a prompt template for each intent. A React UI renders the reply
with react-markdown and shows the detected intent, risk and priority.

## Week 3 — Conversation Memory

Each chat gets its own session ID, and its messages are kept in memory on the server.
Recent history is sent to Groq with every new message, so the assistant can answer
follow-up questions in context. Sessions are kept separate, and New Chat starts a
fresh conversation.
