import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

CONFIG = {
    # =========================
    # Neo4j Configuration
    # =========================
    "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),

    # =========================
    # Ollama Configuration
    # =========================
    "OLLAMA_BASE_URL": os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434"
    ),

    "OLLAMA_MODEL": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest"
    ),

    # =========================
    # CDF-RAG LLM Configuration
    # =========================
    "KNOWLEDGE_REWRITING_MODEL": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest"
    ),

    "QUERY_REFINEMENT_MODEL": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest"
    ),

    "LLM_MODEL": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest"
    ),

    "HALLUCINATION_DETECTION_MODEL": os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:latest"
    ),

    # =========================
    # Embedding Configuration
    # =========================
    "ENCODER_MODEL": os.getenv(
        "EMBEDDING_MODEL",
        "nomic-embed-text:latest"
    ),

    # =========================
    # Pinecone Configuration
    # =========================
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_INDEX": os.getenv("PINECONE_INDEX", "cdf-rag"),
    "PINECONE_HOST": os.getenv("PINECONE_HOST"),

    # =========================
    # Causal Graph
    # =========================
    "CAUSAL_GRAPH_PATH": os.getenv(
        "CAUSAL_GRAPH_PATH",
        "causal_graph.pkl"
    ),
}