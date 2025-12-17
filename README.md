# Multi-Agent Academic Writing System (v0.1)

This project implements a modular multi-agent system for academic coursework writing,
built using CrewAI. The system orchestrates specialized agents responsible for planning,
writing, validating, and editing academic texts, optionally grounded in a shared knowledge base.

## Architecture Overview

The system follows a layered architecture:

- Agents — domain-specific roles (planner, writer, validator, editor)
- Tasks — deterministic pipeline with explicit context passing
- Knowledge — passive document sources (PDF/text)
- Tools — explicit retrieval and parsing utilities
- Crew — orchestration layer

## Agents

- Planner — builds structured outline
- Writer — writes sections using provided context
- Validator — checks factual consistency
- Editor — improves academic style

## Knowledge Handling

In v0.1, knowledge is injected via Crew-level knowledge sources.
This provides background context but does not enforce strict RAG constraints.

Future versions will experiment with:
- explicit retrieval tools
- vector stores (Chroma / Qdrant)
- citation-aware pipelines

## Running the Project

```bash
python main.py
