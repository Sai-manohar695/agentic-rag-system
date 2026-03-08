"""
Tool 1: RAG Search Tool
Embeds documents into Pinecone and retrieves
relevant chunks for user queries.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# ── Config ────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
INDEX_NAME       = "agentic-rag"
EMBED_DIM        = 384  # all-MiniLM-L6-v2 dimension

# ── Init ──────────────────────────────────────────────────
pc      = Pinecone(api_key=PINECONE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name      = INDEX_NAME,
            dimension = EMBED_DIM,
            metric    = "cosine",
            spec      = ServerlessSpec(
                cloud  = "aws",
                region = "us-east-1"
            )
        )
        print(f"Created Pinecone index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)

def ingest_documents(texts: list[str],
                     metadatas: list[dict] = None):
    """Embed and upsert documents into Pinecone."""
    index = get_or_create_index()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 512,
        chunk_overlap = 64
    )

    all_chunks = []
    all_metas  = []

    for i, text in enumerate(texts):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        meta = metadatas[i] if metadatas else {}
        all_metas.extend([meta] * len(chunks))

    # Embed in batches
    embeddings = embedder.encode(
        all_chunks,
        batch_size     = 32,
        show_progress_bar = True
    )

    # Upsert to Pinecone
    vectors = [
        {
            "id"      : f"doc_{i}",
            "values"  : emb.tolist(),
            "metadata": {
                **all_metas[i],
                "text": all_chunks[i]
            }
        }
        for i, emb in enumerate(embeddings)
    ]

    index.upsert(vectors=vectors, batch_size=100)
    print(f"Ingested {len(vectors)} chunks into Pinecone")
    return len(vectors)

def rag_search(query: str, top_k: int = 5) -> str:
    """Search Pinecone for relevant chunks."""
    index = get_or_create_index()

    query_embedding = embedder.encode([query])[0].tolist()

    results = index.query(
        vector          = query_embedding,
        top_k           = top_k,
        include_metadata = True
    )

    if not results['matches']:
        return "No relevant documents found in knowledge base."

    chunks = []
    for i, match in enumerate(results['matches']):
        score = match['score']
        text  = match['metadata'].get('text', '')
        src   = match['metadata'].get('source', 'Unknown')
        chunks.append(
            f"[{i+1}] (score: {score:.3f}) [{src}]\n{text}"
        )

    return "\n\n".join(chunks)

if __name__ == "__main__":
    # Quick test
    test_docs = [
        """Retrieval Augmented Generation (RAG) is a technique
        that combines information retrieval with text generation.
        It retrieves relevant documents from a knowledge base
        and uses them as context for the language model.""",

        """LangChain is a framework for building applications
        powered by language models. It provides tools for
        chaining LLM calls, memory management, and agent
        orchestration.""",

        """Pinecone is a vector database optimized for
        similarity search. It stores embeddings and enables
        fast approximate nearest neighbor search at scale."""
    ]

    metadatas = [
        {"source": "RAG Paper", "topic": "RAG"},
        {"source": "LangChain Docs", "topic": "Framework"},
        {"source": "Pinecone Docs", "topic": "Vector DB"}
    ]

    print("Ingesting test documents...")
    ingest_documents(test_docs, metadatas)

    print("\nSearching for: 'how does RAG work?'")
    result = rag_search("how does RAG work?")
    print(result)