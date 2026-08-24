"""
Enterprise AI Customer Support Assistant -- Live Demo (Group 1)
Standalone Gradio app for Hugging Face Spaces.

This is the notebook's logic (Insurance_RAG_Demo.ipynb) ported to a single script so it can
run as an always-on Space with a persistent public URL, instead of Colab's temporary
gradio.live share link.

Reads the LLM provider key from the Space's Secrets (Settings -> Variables and secrets),
never from code -- see README.md for setup instructions.
"""

import os
import re
import time
import random

import numpy as np
import faiss
import gradio as gr
import openai
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Step 0 -- Provider setup (reads from Space Secrets, set in the HF UI)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. In your Space, go to Settings -> Variables and secrets -> "
        "New secret, name it GROQ_API_KEY, and paste your key from console.groq.com/keys."
    )

client = openai.OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)
GEN_MODEL = "openai/gpt-oss-20b"  # Groq's current recommended fast/free-tier model
# If this 404s later (providers deprecate models periodically), check
# console.groq.com/docs/models and update GEN_MODEL.

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Step 2 -- Sample policy/FAQ documents (Section 3.1 ingestion sources)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "doc_id": "POL-HOME-001",
        "title": "Homeowners Policy — Water Damage Coverage",
        "text": (
            "Section 4.2 Water Damage: Sudden and accidental water discharge from a plumbing, "
            "heating, or air-conditioning system (e.g., a burst pipe) is covered under Section 4.2, "
            "subject to the policy deductible of $1,000. Damage caused by gradual leakage, seepage, "
            "or lack of maintenance over a period exceeding 14 days is excluded under Section 4.5 "
            "(Gradual Damage Exclusion). Flood damage from external sources is not covered under this "
            "policy and requires separate flood insurance."
        ),
    },
    {
        "doc_id": "POL-HOME-002",
        "title": "Homeowners Policy — Deductibles",
        "text": (
            "Section 2.1 Deductibles: The standard deductible for all covered perils under this "
            "homeowners policy is $1,000 per claim, unless a higher wind/hail deductible of 2% of "
            "dwelling coverage applies in designated coastal zones, as shown on the declarations page."
        ),
    },
    {
        "doc_id": "POL-CLAIMS-003",
        "title": "Claims Procedure — Filing a Water Damage Claim",
        "text": (
            "To file a water damage claim: (1) Stop the source of water if safe to do so. "
            "(2) Document damage with photos. (3) Submit a claim via the app, portal, or by calling "
            "the claims line. (4) An adjuster will contact you within 2 business days. (5) Approved "
            "claims are typically paid within 10-15 business days after adjuster sign-off."
        ),
    },
    {
        "doc_id": "POL-CLAIMS-004",
        "title": "Claims Procedure — Status Definitions",
        "text": (
            "Claim status values: 'Submitted' (received, not yet reviewed), 'Under Review' (adjuster "
            "assigned), 'Additional Info Requested' (documentation needed from policyholder), "
            "'Approved' (payment being processed), 'Denied' (see denial letter for reason and appeal "
            "rights), 'Closed' (claim resolved)."
        ),
    },
    {
        "doc_id": "POL-FAQ-005",
        "title": "FAQ — Renewals",
        "text": (
            "Policies renew automatically 30 days before the expiration date unless cancelled in "
            "writing. Renewal premiums may change based on updated risk factors, claims history, and "
            "regional rate filings. Customers are notified of renewal premium changes at least 30 days "
            "in advance by mail and email."
        ),
    },
    {
        "doc_id": "POL-FAQ-006",
        "title": "FAQ — Premium Calculation Factors",
        "text": (
            "Premiums are calculated based on dwelling replacement cost, location risk (flood zone, "
            "wildfire zone, crime rate), coverage limits selected, deductible chosen, claims history, "
            "and applicable discounts (multi-policy, security system, claims-free)."
        ),
    },
    {
        "doc_id": "POL-COMPLIANCE-007",
        "title": "Compliance — Required Disclosures",
        "text": (
            "All coverage explanations provided to customers must include the disclaimer: 'This "
            "explanation is for informational purposes only and does not modify or override the terms "
            "of your official policy document. Please refer to your declarations page and policy "
            "contract for binding coverage terms.' Binding coverage determinations may only be made "
            "by a licensed claims adjuster."
        ),
    },
    {
        "doc_id": "POL-FAQ-008",
        "title": "FAQ — Escalation to a Human Agent",
        "text": (
            "Customers should be connected to a licensed human agent for: formal complaints, suspected "
            "fraud, coverage disputes, requests for legal or investment advice, vulnerable-customer "
            "situations, or any question the assistant cannot answer with confidence from approved "
            "documentation."
        ),
    },
]

# ---------------------------------------------------------------------------
# Step 3 -- Ingestion: chunking, PII scrub, embeddings, vector index
# ---------------------------------------------------------------------------
def scrub_pii(text: str) -> str:
    """Lightweight stand-in for the production PII/PHI redaction step (Section 3.1 / 7.2)."""
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", text)
    text = re.sub(r"\b\d{16}\b", "[REDACTED-CARD]", text)
    return text


def chunk_document(doc, max_chars=350):
    """Structure-aware-ish chunking: split on sentence boundaries, pack up to max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", doc["text"].strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current += (" " if current else "") + s
    if current:
        chunks.append(current.strip())
    return chunks


records = []  # each: {doc_id, title, chunk_id, text}
for doc in DOCUMENTS:
    clean_text = scrub_pii(doc["text"])
    for i, chunk in enumerate(chunk_document({**doc, "text": clean_text})):
        records.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "chunk_id": f'{doc["doc_id"]}-{i}',
            "text": chunk,
        })

texts = [r["text"] for r in records]
embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype="float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity via inner product on normalized vectors
index.add(embeddings)

print(f"Vector index built: {index.ntotal} vectors of dimension {dimension}.")

# ---------------------------------------------------------------------------
# Step 4 -- RAG query-time pipeline
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 3):
    """Hybrid-retrieval stand-in: dense vector search (demo omits BM25/rerank/knowledge-graph
    expansion from Section 3.2 for simplicity, but the interface is the same)."""
    q_emb = EMBED_MODEL.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        r = records[idx]
        results.append({**r, "score": float(score)})
    return results


SYSTEM_PROMPT = """You are a customer support assistant for a global insurance carrier.
Rules you must follow:
1. Answer ONLY using the provided context passages. Do not use outside knowledge.
2. You MUST cite the source of every factual claim by including its doc_id exactly as written,
   in square brackets, e.g. [POL-HOME-001]. Every answer must contain at least one citation.
3. If the context does not contain the answer, say so plainly and do not guess.
4. Never give legal, medical, or investment advice. Never make a binding coverage determination.
5. Always include the standard disclaimer if you explain coverage.

Example of a correctly formatted answer:
"Water damage from a burst pipe is covered under Section 4.2 [POL-HOME-001], subject to your
$1,000 deductible [POL-HOME-002]. This is for informational purposes only; see your policy
contract for binding terms."
"""


def build_prompt(query, passages):
    context_block = "\n\n".join(
        f'[{p["doc_id"]}] ({p["title"]}): {p["text"]}' for p in passages
    )
    doc_ids = ", ".join(p["doc_id"] for p in passages)
    return f"""Context passages:
{context_block}

Customer question: {query}

Answer the question using only the context above. You MUST include at least one citation from
this exact list, in square brackets: {doc_ids}"""


def generate_answer(query, passages, extra_instruction=None, max_retries=4, base_delay=1.5):
    """Calls the LLM with retry-with-backoff so a transient 429/5xx doesn't crash the demo
    (Section 9.2 'graceful degradation' in miniature -- retry before failing/escalating).
    `extra_instruction` lets answer_query() ask for a citation-repair pass."""
    prompt = build_prompt(query, passages)
    if extra_instruction:
        prompt += f"\n\n{extra_instruction}"
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GEN_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content

        except openai.RateLimitError as e:
            last_error = e
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[retry {attempt + 1}/{max_retries}] Rate limited (429). Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except (openai.APIConnectionError, openai.InternalServerError) as e:
            last_error = e
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[retry {attempt + 1}/{max_retries}] Transient error "
                  f"({type(e).__name__}). Retrying in {wait:.1f}s...")
            time.sleep(wait)

        except openai.NotFoundError as e:
            raise RuntimeError(
                f"Model '{GEN_MODEL}' was not found by the provider (it may have been deprecated). "
                "Check console.groq.com/docs/models and update GEN_MODEL in app.py."
            ) from e

        except openai.AuthenticationError as e:
            raise RuntimeError(
                "API authentication failed -- the GROQ_API_KEY secret is missing, invalid, or "
                "expired. Check Settings -> Variables and secrets on this Space."
            ) from e

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts ({type(last_error).__name__}: {last_error})."
    )


# ---------------------------------------------------------------------------
# Step 5 -- Guardrails: confidence gate, citation check, citation-repair retry
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.35  # cosine similarity floor -- tune with your golden set (Section 6.1)


def faithfulness_check(answer_text, passages):
    """Stand-in for Section 3.2's faithfulness/citation verification step: require that at
    least one retrieved doc_id is actually cited in the answer."""
    cited = [p["doc_id"] for p in passages if p["doc_id"] in answer_text]
    return len(cited) > 0, cited


def answer_query(query: str, top_k: int = 3):
    """Full grounded-RAG pipeline with guardrails."""
    passages = retrieve(query, top_k=top_k)
    top_score = passages[0]["score"] if passages else 0.0

    if top_score < CONFIDENCE_THRESHOLD:
        return {
            "answer": None,
            "status": "ESCALATE_LOW_CONFIDENCE",
            "reason": f"Top retrieval similarity {top_score:.2f} below threshold {CONFIDENCE_THRESHOLD}.",
            "citations": [],
            "passages": passages,
        }

    try:
        raw_answer = generate_answer(query, passages)
    except RuntimeError as e:
        return {
            "answer": None,
            "status": "ESCALATE_LLM_UNAVAILABLE",
            "reason": str(e),
            "citations": [],
            "passages": passages,
        }

    clean_answer = scrub_pii(raw_answer)
    grounded, cited_docs = faithfulness_check(clean_answer, passages)

    if not grounded:
        repair_instruction = (
            "Your previous answer did not include a citation in square brackets. "
            f'Here is your previous answer: "{clean_answer}" '
            "Rewrite it, keeping the same content, but add at least one citation from the "
            "provided doc_id list in square brackets, e.g. [POL-HOME-001]."
        )
        try:
            repaired = generate_answer(query, passages, extra_instruction=repair_instruction)
            repaired = scrub_pii(repaired)
            grounded, cited_docs = faithfulness_check(repaired, passages)
            if grounded:
                clean_answer = repaired
        except RuntimeError:
            pass

    if not grounded:
        return {
            "answer": None,
            "status": "ESCALATE_NO_CITATION",
            "reason": "Generated answer did not cite any retrieved source, even after a repair "
                      "attempt; suppressing per Section 3.2.",
            "citations": [],
            "passages": passages,
        }

    return {
        "answer": clean_answer,
        "status": "ANSWERED",
        "reason": None,
        "citations": cited_docs,
        "passages": passages,
    }


# ---------------------------------------------------------------------------
# Step 6 -- Escalation workflow (Section 4.4 Path C)
# ---------------------------------------------------------------------------
ESCALATION_KEYWORDS = [
    "complaint", "complain", "fraud", "sue", "lawsuit", "lawyer", "legal advice",
    "investment advice", "cancel my policy immediately", "discriminat", "unfair",
]


def detect_escalation_intent(query: str):
    q = query.lower()
    return [kw for kw in ESCALATION_KEYWORDS if kw in q]


def handle_customer_message(query: str):
    """Top-level orchestrator: Section 4.1's supervisor/router logic in miniature."""
    intent_hits = detect_escalation_intent(query)
    if intent_hits:
        return {
            "path": "PATH_C_HUMAN_REQUIRED",
            "trigger": f"Keyword match: {intent_hits}",
            "message_to_customer": (
                "I'm connecting you with a licensed support agent who can help with this right "
                "away -- they'll have full context on what you've told me so far."
            ),
        }

    result = answer_query(query)
    if result["status"] == "ANSWERED":
        return {
            "path": "PATH_A_AUTOMATED",
            "trigger": None,
            "answer": result["answer"],
            "citations": result["citations"],
        }
    else:
        return {
            "path": "PATH_C_HUMAN_REQUIRED",
            "trigger": result["reason"],
            "message_to_customer": (
                "I want to make sure you get an accurate answer here -- let me connect you with "
                "a specialist who can look into this."
            ),
        }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def gradio_handler(message, history):
    r = handle_customer_message(message)
    if r["path"] == "PATH_A_AUTOMATED":
        cites = ", ".join(r["citations"])
        return f'{r["answer"]}\n\n_Sources: {cites}_'
    else:
        return f'{r["message_to_customer"]}\n\n_[Internal: escalation trigger -- {r["trigger"]}]_'


demo = gr.ChatInterface(
    gradio_handler,
    title="Enterprise AI Customer Support Assistant -- Live Demo (Group 1)",
    description=(
        "RAG-grounded insurance support assistant with citations and human escalation. "
        "Demo build for the Enterprise AI System Architect Program capstone."
    ),
    examples=[
        "Is water damage covered under my policy?",
        "What's my deductible?",
        "How long does it take to get a claim paid?",
        "I want to file a complaint about my claim.",
    ],
)

if __name__ == "__main__":
    # Render (and most PaaS free tiers) assign a port via the PORT env var and expect the
    # app to bind to 0.0.0.0, not localhost. Defaults to Gradio's usual 7860 for local runs.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
