
import re

from llama_index.core.base.llms.types import CompletionResponse


import sqlite3
from datetime import datetime, timezone

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

from config import (
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_SIMILARITY_SCORE,
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT_TEMPLATE,
    TOKEN_LIMIT,
    CHAT_DB_PATH,
    SIMILARITY_TOP_K,
    MAX_HISTORY_MESSAGES,
    FOLLOW_UP_QUESTIONS_PROMPT_TEMPLATE,
    ANSWER_VERIFICATION_PROMPT_TEMPLATE,
)


def verify_answer(question, answer, context):
    """Remove claims that are not supported by retrieved context."""

    prompt = ANSWER_VERIFICATION_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        answer=answer,
    )

    try:
        response = llm.complete(prompt)
        verified = response.text.strip()

        if verified:
            return verified

        return "I don't know based on the available documents."

    except Exception as exc:
        print(f"Answer verification failed: {exc}")
        return (
            "I couldn't verify the answer against the available "
            "documents, so I won't present it as verified."
        )


def generate_follow_up_questions(
    question,
    answer,
    context,
    history_text,
):
    """Generate three follow-up questions from retrieved context."""

    prompt = FOLLOW_UP_QUESTIONS_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        context=context,
        history=history_text or "No previous conversation.",
    )

    try:
        response = llm.complete(prompt)
        suggestions = []

        for line in response.text.strip().splitlines():
            cleaned = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                line,
            ).strip().strip('"').strip("'")

            if (
                cleaned
                and cleaned not in suggestions
                and cleaned.endswith("?")
            ):
                suggestions.append(cleaned)

        return suggestions[:3]

    except Exception as exc:
        print(f"Could not generate follow-up questions: {exc}")
        return []





llm = Ollama(
    model="llama3.2:latest",
    request_timeout=120.0,
    context_window=4096,
)


def initialize_chat_db():
    """Create the persistent chat history table if needed."""
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_message(role, content):
    """Save one user or assistant message to SQLite."""
    created_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (role, content, created_at)
            VALUES (?, ?, ?)
            """,
            (role, content, created_at),
        )


def load_recent_history():
    """Load recent messages in chronological order."""
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (MAX_HISTORY_MESSAGES,),
        ).fetchall()

    # SQL returns newest first; memory needs oldest first.
    return list(reversed(rows))


def create_memory():
    """Restore recent persisted messages into LlamaIndex memory."""
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=TOKEN_LIMIT
    )

    for role, content in load_recent_history():
        if role == "user":
            message_role = MessageRole.USER
        elif role == "assistant":
            message_role = MessageRole.ASSISTANT
        else:
            continue

        memory.put(
            ChatMessage(
                role=message_role,
                content=content,
            )
        )

    return memory


def format_history(messages):
    """Convert chat messages into readable prompt history."""
    return "\n".join(
        f"{message.role.value}: {message.content}"
        for message in messages
    )


def rewrite_question(question, history_text):
    """Resolve follow-up references before document retrieval."""
    if not history_text.strip():
        return question

    try:
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
            history=history_text,
            question=question,
        )

        response = llm.complete(prompt)
        rewritten = response.text.strip()

        # Avoid using an empty rewrite.
        return rewritten if rewritten else question

    except Exception as exc:
        print(f"Query rewriting failed: {exc}")
        print("Continuing with the original question.")
        return question


def retrieve_relevant_documents(retriever, search_question):
    """Retrieve chunks and filter out low-relevance results."""
    results = retriever.retrieve(search_question)

    print("\nRetrieved candidates:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result.score} | "
            f"source={result.node.metadata.get('file_name', 'Unknown')}"
        )


    relevant_results = [
        result
        for result in results
        if result.score is not None
        and result.score >= MIN_SIMILARITY_SCORE
    ]

    print(f"Similarity threshold: {MIN_SIMILARITY_SCORE}")
    print(f"Candidates: {len(results)}")
    print(f"Relevant chunks: {len(relevant_results)}")

    return relevant_results


def build_context(results):
    """Label each context chunk so the LLM can cite its source."""
    context_parts = []
    source_details = []

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Chunk {rank} ---")
        print(result.node.get_content())
        filename = result.node.metadata.get(
            "file_name", "Unknown source"
        )

        label = f"Source {rank}"

        context_parts.append(
            f"[{label}: {filename}]\n"
            f"{result.node.get_content()}"
        )

        source_details.append({
            "label": label,
            "filename": filename,
            "score": result.score,
        })

    return "\n\n".join(context_parts), source_details


def generate_answer(question, history_text, context):
    """Generate an answer grounded in retrieved document context."""
    prompt = RAG_PROMPT_TEMPLATE.format(
        history=history_text or "No previous conversation.",
        context=context,
        question=question,
    )

    response = llm.complete(prompt)
    return response.text.strip()


def main():
    initialize_chat_db()
    memory = create_memory()

    print("Persistent chat history loaded.")
    print(f"History database: {CHAT_DB_PATH}")

    # 1. Connect to the existing vector database.
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            "Index not found. Run 'python ingest.py' first."
        ) from exc

    print(f"Indexed chunks available: {collection.count()}")

    if collection.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Run 'python ingest.py'."
        )

    # 2. Load the same embedding model used during ingestion.
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # 3. Connect LlamaIndex to ChromaDB.
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # 4. Create the retriever.
    retriever = index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K
    )

    # 5. Interactive question loop.
    last_suggestions = []

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("Conversation saved.")
            break

        if not question:
            continue

        # Allow the user to select a previously suggested question by number.
        if question.isdigit():
            choice = int(question)
            if 1 <= choice <= len(last_suggestions):
                question = last_suggestions[choice - 1]
                print(f"Selected question: {question}")
            else:
                print(
                    f"Invalid selection. Enter 1–{len(last_suggestions)} "
                    "or type a question."
                )
                continue

        # Build history before rewriting the question.
        chat_history = memory.get()
        history_text = format_history(chat_history)

        # 6. Rewrite follow-up questions BEFORE retrieval.
        search_question = rewrite_question(
            question,
            history_text,
        )

        print(f"\nOriginal question: {question}")
        print(f"Search question: {search_question}")

        # 7. Retrieve and filter relevant document chunks.
        try:
            relevant_results = retrieve_relevant_documents(
                retriever,
                search_question,
            )
        except Exception as exc:
            print(f"Document retrieval failed: {exc}")
            continue

        if not relevant_results:
            answer = (
                "I don't know based on the available documents. "
                "No sufficiently relevant document chunks were found."
            )

            print(f"\nAnswer: {answer}")

            # Persist even unanswered turns so the chat is continuous.
            memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=question,
                )
            )
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                )
            )

            try:
                save_message("user", question)
                save_message("assistant", answer)
            except sqlite3.Error as exc:
                print(f"Could not save chat history: {exc}")

            continue

        # 8. Build source-labelled context.
        context, sources = build_context(relevant_results)

        # 9. Generate a source-grounded answer.
        try:
            draft_answer = generate_answer(
                question=question,
                history_text=history_text,
                context=context,
            )
            print("\n--- DRAFT ANSWER ---")
            print(draft_answer)
            verified_answer: str = verify_answer(
            question=question,
            answer=draft_answer,
            context=context,
            )
            print("\n--- VERIFIED ANSWER ---")
            print(verified_answer)
            answer = verified_answer
            print("\nAnswer:")
            print(answer)

            print("\nSources retrieved:")
            for source in sources:
                print(
                    f"[{source['label']}] "
                    f"{source['filename']} "
                    f"(score: {source['score']:.3f})"
                )

            # 10. Update in-memory and persistent history.
            memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=question,
                )
            )
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=answer,
                )
            )

            try:
                save_message("user", question)
                save_message("assistant", answer)
            except sqlite3.Error as exc:
                print(f"Could not save chat history: {exc}")

            # 11. Generate and display suggested follow-up questions.
            suggestions = generate_follow_up_questions(
                question=question,
                answer=answer,
                context=context,
                history_text=history_text,
            )

            if suggestions:
                print("\nSuggested next questions:")
                for number, suggestion in enumerate(suggestions, start=1):
                    print(f"  {number}. {suggestion}")
                print(
                    "\nTip: Enter 1, 2, or 3 at the next prompt "
                    "to ask a suggested question."
                )

            last_suggestions = suggestions

        except Exception as exc:
            print(f"\nAnswer generation failed: {exc}")
            print(
                "Check that Ollama is running and "
                "llama3.2:latest is installed."
            )


if __name__ == "__main__":
    main()
