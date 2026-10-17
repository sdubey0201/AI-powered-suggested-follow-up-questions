
"""
ingest.py — Document ingestion pipeline for the RAG Assistant.

Run this script once before querying, and re-run it whenever you
add or update files in the documents/ folder.

What it does:
  1. Loads all .md files from DOCUMENT_DIR
  2. Splits them into overlapping token chunks
  3. Converts each chunk to a 384-dim embedding vector (locally)
  4. Saves all vectors to a persistent ChromaDB collection on disk

Usage:
  python ingest.py
"""

from pathlib import Path

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    DOCUMENT_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def main():
    # ── Step 1: Load source documents ────────────────────────────────────────
    # SimpleDirectoryReader walks DOCUMENT_DIR and loads every .md file.
    # required_exts filters out any non-Markdown files (images, PDFs, etc.).
    # Each file becomes one Document object with its text and file metadata.
    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENT_DIR),
        required_exts=[".md"],
    ).load_data()

    print(f"Loaded documents: {len(documents)}")

    # Guard: if the folder is empty or has no .md files, fail early with a
    # clear message rather than silently creating an empty vector store.
    if not documents:
        raise RuntimeError("No Markdown documents found.")

    # ── Step 2: Split documents into chunks ──────────────────────────────────
    # SentenceSplitter breaks each document into overlapping token windows.
    # chunk_size=128  → each chunk is at most 128 tokens long
    # chunk_overlap=40 → last 40 tokens of chunk N are the first 40 of chunk N+1
    #                    this ensures sentences at boundaries are not lost
    # Output: a list of Node objects, each holding one chunk of text + metadata
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Created chunks: {len(nodes)}")

    # Print every chunk so you can inspect how the text was split.
    # Useful for tuning CHUNK_SIZE and CHUNK_OVERLAP in config.py.
    for i, node in enumerate(nodes, start=1):
        print(f"\n--- Chunk {i} ---")
        print(node.text)

    # ── Step 3: Load the embedding model ─────────────────────────────────────
    # HuggingFaceEmbedding downloads and caches the model locally on first run.
    # It converts each chunk of text into a 384-dimensional float vector.
    # IMPORTANT: this must be the same model used in query.py — the vectors
    # stored here can only be compared against queries embedded by the same model.
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )

    # ── Step 4: Set up persistent ChromaDB ───────────────────────────────────
    # PersistentClient writes the vector store to disk at DB_DIR (chroma_db/).
    # The data survives between runs — query.py reads from the same folder.
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete the existing collection (if any) before re-creating it.
    # This ensures a clean rebuild on every run — no stale or duplicate vectors
    # accumulate if you re-ingest after editing your documents.
    # ValueError is raised if the collection does not exist yet — safe to ignore.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass

    # Create a fresh, empty collection to hold the new vectors.
    collection = client.create_collection(name=COLLECTION_NAME)

    # ChromaVectorStore adapts ChromaDB's collection to the LlamaIndex
    # VectorStore interface so VectorStoreIndex can write to it.
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    # StorageContext bundles the vector store so VectorStoreIndex knows
    # where to persist the embeddings it generates.
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # ── Step 5: Generate embeddings and store them ───────────────────────────
    # VectorStoreIndex loops over every node, calls embed_model to produce a
    # vector for each chunk, and writes the (vector, text, metadata) triplet
    # to ChromaDB via the storage_context.
    # After this line, chroma_db/ on disk contains all indexed vectors.
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("\nIngestion completed successfully.")
    # collection.count() confirms how many vectors are now stored in ChromaDB.
    print(f"Indexed chunks: {collection.count()}")
    print(f"Database location: {DB_DIR.resolve()}")


if __name__ == "__main__":
    main()
