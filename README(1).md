---
title: Insurance RAG Demo
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Enterprise AI Customer Support Assistant — Live Demo (Group 1)

RAG-grounded insurance support assistant with mandatory citations, a confidence-gated
faithfulness guardrail, and keyword + low-confidence escalation to a simulated human agent.
Built for the TEKsystems Enterprise AI System Architect Program capstone (Case Study 1).

## Setup (one-time)

This Space needs one secret to run:

1. Go to **Settings → Variables and secrets → New secret**.
2. Name: `GROQ_API_KEY`
3. Value: your key from [console.groq.com/keys](https://console.groq.com/keys) (free tier, no
   credit card required).
4. Save — the Space will rebuild automatically and come online within a minute or two.

## What this demonstrates

- **Ingestion:** structure-aware chunking + PII scrub + TF-IDF vectorization (scikit-learn).
  This deploy uses TF-IDF instead of neural embeddings deliberately -- `sentence-transformers`
  + PyTorch exceeds the 512MB RAM limit on free hosting tiers (Render free web services and
  similar). See `Insurance_RAG_Demo.ipynb` for the neural-embedding + FAISS version if you're
  running somewhere with more memory headroom (e.g., Colab).
- **RAG:** retrieval → grounded prompt assembly → generation via Groq's `openai/gpt-oss-20b`.
- **Guardrails:** a confidence gate refuses to answer below a similarity threshold instead of
  guessing; a citation check suppresses any answer that doesn't cite a retrieved source, with
  one automatic repair attempt before escalating.
- **Escalation:** complaint/fraud/legal-advice keywords, or a failed guardrail, route to a
  simulated human handoff — nothing fails silently.

See the companion notebook `Insurance_RAG_Demo.ipynb` and
`Hands-On_Demo_Implementation_Guide.md` for the full walkthrough mapped to the proposal
sections this demo illustrates.
