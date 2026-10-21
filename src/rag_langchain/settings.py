"""All project settings and one ready-made LangChain embeddings object."""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings


DATA_DIR = Path("data")
CHROMA_DIR = Path("storage/langchain_chroma")
COLLECTION_NAME = "langchain_demo"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
OLLAMA_MODEL = "qwen3:4b-instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_BATCH_SIZE = 1000


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Return LangChain's ready adapter around SentenceTransformers.
    """

    return HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda", "local_files_only": True},
        encode_kwargs={
            "prompt": "passage: ",
            "batch_size": 16,
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "prompt": "query: ",
            "normalize_embeddings": True,
        },
        show_progress=True
    )
