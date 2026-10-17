"""
config.py — Central configuration for the RAG Assistant.

All tuneable values live here. No magic numbers should appear in
ingest.py or query.py — import from this file instead.
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────

# BASE_DIR: absolute path to the folder containing this file.
# Using __file__ makes all paths work regardless of where you run the script from.
BASE_DIR = Path(__file__).resolve().parent

# DOCUMENT_DIR: folder scanned by ingest.py for .md knowledge base files.
# Drop new Markdown documents here, then re-run ingest.py to index them.
DOCUMENT_DIR = BASE_DIR / "documents"

# DB_DIR: on-disk ChromaDB storage created automatically by ingest.py.
# Do not edit this folder manually — it is managed entirely by ChromaDB.
DB_DIR = BASE_DIR / "chroma_db"

# ── Vector database ───────────────────────────────────────────────────────────

# COLLECTION_NAME: the logical bucket inside ChromaDB that holds all vectors.
# ingest.py writes to this collection; query.py reads from it.
# Changing this name requires a full re-ingestion.
COLLECTION_NAME = "kubernetes_docs"

# ── Embedding model ───────────────────────────────────────────────────────────

# EMBEDDING_MODEL: HuggingFace model used to convert text → 384-dim float vectors.
# Must be the same model in both ingest.py (indexing) and query.py (querying).
# If you change this, delete chroma_db/ and re-run ingest.py — old vectors
# were produced by a different model and will give wrong similarity scores.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking configuration ────────────────────────────────────────────────────

# CHUNK_SIZE: maximum number of tokens per chunk fed to the embedding model.
# Smaller = more precise retrieval but more chunks to store.
# Larger = more context per chunk but less retrieval precision.
CHUNK_SIZE = 128

# CHUNK_OVERLAP: how many tokens the end of one chunk shares with the start
# of the next. Prevents context from being cut mid-sentence at boundaries.
CHUNK_OVERLAP = 40

# ── Retrieval configuration ───────────────────────────────────────────────────

# SIMILARITY_TOP_K: how many candidate chunks ChromaDB returns per query.
# These are the top-k closest vectors by cosine similarity before filtering.
# Retrieval settings
SIMILARITY_TOP_K = 3
MAX_HISTORY_MESSAGES = 20


# MIN_SIMILARITY_SCORE: quality gate applied after retrieval.
# Any chunk with a cosine similarity score below this value is discarded.
# Range 0.0–1.0. Raise for stricter precision; lower for broader recall.
MIN_SIMILARITY_SCORE = 0.35


# ── Conversation memory ───────────────────────────────────────────────────────

# TOKEN_LIMIT: maximum number of tokens kept in the ChatMemoryBuffer.
# When the buffer exceeds this limit, the oldest messages are dropped automatically.
# Prevents the conversation history from overflowing the LLM's context window (4096).
TOKEN_LIMIT = 2000

# ── Query rewriting prompt ────────────────────────────────────────────────────

# QUERY_REWRITE_PROMPT_TEMPLATE: sent to the LLM before retrieval when conversation
# history exists. Its job is to rewrite ambiguous follow-up questions into fully
# self-contained standalone questions so ChromaDB can retrieve the right chunks.
# Example: "What about its limits?" → "What are the resource limits of a Kubernetes pod?"
# Placeholders filled at runtime:
#   {history}  — formatted conversation history from ChatMemoryBuffer
#   {question} — the user's current follow-up question
# QUERY_REWRITE_PROMPT_TEMPLATE = """
# Your only job is to resolve pronouns and implicit references in the current question using the conversation history.

# Rules:
# - If the question contains no pronouns or implicit references (e.g. "it", "they", "that", "its", "their", "this"), return it EXACTLY as written — do not rephrase, paraphrase, or add detail.
# - If the question contains a pronoun or implicit reference, replace only that reference with the specific subject from the conversation history. Keep all other words identical.
# - Do not answer the question.
# - Do not add facts, context, or keywords not present in the original question.
# - Return only the final question, no explanation.

# Conversation history:
# {history}

# Current question:
# {question}

# Standalone search question:
# """


QUERY_REWRITE_PROMPT_TEMPLATE = """
Task: Rewrite the CURRENT QUESTION into one standalone search question.

Use previous messages only to resolve missing references.

STRICT RULES:
- Your output must be exactly one question.
- Keep the meaning of the CURRENT QUESTION unchanged.
- Never answer the question.
- Never copy or summarize an earlier assistant answer.
- Never include source citations or document text.
- If the current question is already clear, return it unchanged.
- Return only the rewritten question.

Previous conversation:
{history}

CURRENT QUESTION:
{question}

Standalone search question:
"""


# Persistent conversation storage
CHAT_DB_PATH = BASE_DIR / "chat_history.db"

# ── Answer generation prompt ──────────────────────────────────────────────────

# RAG_PROMPT_TEMPLATE: the instruction string sent to the LLM for answer generation.
# Placeholders filled at runtime:
#   {context}  — concatenated text of all chunks that passed MIN_SIMILARITY_SCORE
#   {question} — the user's original question
# The "do not invent facts" rule is critical — it prevents the LLM from hallucinating
# information that is not present in the retrieved context.
# RAG_PROMPT_TEMPLATE = """
# You are a technical knowledge assistant.

# Answer the user's question using ONLY the provided document context.

# Rules:
# 1. Treat document context as the source of factual information.
# 2. Do not invent facts or rely on general knowledge to fill gaps.
# 3. Conversation history may help interpret the question, but it is NOT
#    an authoritative source of technical facts.
# 4. Cite factual claims using the source labels provided in the context,
#    for example [Source 1].
# 5. Only cite source labels that actually appear in the provided context.
# 6. If the context does not contain enough information, clearly say:
#    "I don't know based on the available documents."
# 7. Be clear and concise. Do not claim that a source supports information
#    that it does not contain.

# Conversation history:
# {history}

# Document context:
# {context}

# Question:
# {question}

# Answer with source citations:
# """

# RAG_PROMPT_TEMPLATE = """
# You are a technical knowledge assistant.

# Answer the question using ONLY facts explicitly supported by
# the provided document context.

# STRICT RULES:
# 1. Every factual claim must be supported by the context.
# 2. Do not use general knowledge to fill gaps.
# 3. Do not infer benefits, advantages, disadvantages, or implications
#    unless the context explicitly supports them.
# 4. Cite each factual claim using the exact source label, such as [Source 1].
# 5. Never invent source labels or citations.
# 6. If the context only partially answers the question, state what
#    is supported and clearly identify what is not documented.
# 7. If the context does not answer the question, say:
#    "I don't know based on the available documents."
# 8. Do not contradict yourself.
# 9. Do not repeat the question unnecessarily.

# Conversation history (use only to understand references):
# {history}

# Retrieved document context:
# {context}

# Question:
# {question}

# Answer:
# """


RAG_PROMPT_TEMPLATE = """
You answer technical questions using the supplied documents.

Rules:
1. Answer using facts explicitly stated in the context.
2. Do not add outside knowledge or unsupported conclusions.
3. Cite supporting sources using their exact labels, such as [Source 1].
4. If the answer is explicitly stated, answer directly.
5. If the context does not contain the answer, say:
   I don't know based on the available documents.
6. Do not reject a directly stated fact merely because the
   question uses different wording.

Conversation history:
{history}

Retrieved context:
{context}

Question:
{question}

Answer:
"""



FOLLOW_UP_QUESTIONS_PROMPT_TEMPLATE = """
You are a learning assistant helping a user explore technical documents.

Generate exactly 3 useful follow-up questions based on:
- The user's current question
- The answer just generated
- The available document context
- The conversation history

Rules:
1. Suggest questions that naturally follow from the current topic.
2. Prefer questions answerable using the supplied document context.
3. Do not repeat the current question or questions already asked.
4. Include a mix of deeper concepts, practical usage, and related topics
   when supported by the context.
5. Do not answer the suggested questions.
6. Return exactly 3 questions, one per line.
7. Do not include numbering, bullets, or explanations.

Current question:
{question}

Answer:
{answer}

Document context:
{context}

Conversation history:
{history}

Suggested next questions:
"""


ANSWER_VERIFICATION_PROMPT_TEMPLATE = """
You are a strict evidence verifier for a retrieval-augmented
question-answering system.

Compare the draft answer with the retrieved document context.

Rules:
- Accept only claims directly supported by the context.
- Reject unsupported claims, assumptions, and outside knowledge.
- Treat source citations as valid only when their labels exist in context.
- If only part of the answer is supported, keep only that part.
- If no meaningful answer is supported, return exactly:
  I don't know based on the available documents.
- Return the corrected answer only.
- Do not add new facts.
Additional verification rules:

- Shared networking or storage does not prove anything about lifecycle
  independence unless the context explicitly states the relationship.
- A statement about what is possible, supported, or required must be
  explicitly supported by the retrieved context.
- Do not answer "yes" or "no" merely because the context discusses
  a related concept.
- Reject conclusions based on "this implies", "therefore", or similar
  reasoning unless the conclusion itself is supported by the context.
- If the evidence is insufficient, return exactly:
  I don't know based on the available documents.


Question:
{question}

Retrieved context:
{context}

Draft answer:
{answer}

Verified answer:
"""

