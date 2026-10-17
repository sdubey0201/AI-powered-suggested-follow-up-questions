
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

DOCUMENT_DIR = Path("documents")
DB_DIR = Path("chroma_db")


def main():
    # 1. Load documents
    documents = SimpleDirectoryReader(
        input_dir=str(DOCUMENT_DIR),
        required_exts=[".md"],
    ).load_data()

    print(f"Loaded documents: {len(documents)}")

    # 2. Split documents into chunks
    # question  why hard coding here what if document is too big and 200 token is too bing or not enough ? 
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=30,
    )
    # question node lenght is only 1 document lenght is 1 since i know that only 1 document is present in directory.
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created chunks: {len(nodes)}")

    # 3. Initialize a local embedding model
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create a persistent ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name="kubernetes_docs"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5. Index document chunks
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 6. Retrieve the most relevant chunks
    retriever = index.as_retriever(similarity_top_k=3)

    question = "Why can a Kubernetes pod remain in Pending state?"
    results = retriever.retrieve(question)

    print(f"\nQuestion: {question}")
    print(f"\nRetrieved results: {results}")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Similarity score: {result.score}")
        print(result.text)


if __name__ == "__main__":
    main()
