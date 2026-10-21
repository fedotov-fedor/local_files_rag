"""Read data/, split documents and rebuild the local Chroma index."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_langchain.settings import (
    CHROMA_BATCH_SIZE,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    create_embeddings
)


# LangChain loaders turn every file (or PDF page) into a Document.
documents = DirectoryLoader(
    str(DATA_DIR),
    glob="**/*.pdf", 
    loader_cls=PyMuPDFLoader, 
    show_progress=True
).load()

for extension in ("txt", "md"):
    documents += DirectoryLoader(
        str(DATA_DIR),
        glob=f"**/*.{extension}",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True
    ).load()

# Keep short relative paths in Chroma metadata. PyMuPDF numbers pages from zero.
for document in documents:
    document.metadata["source"] = Path(document.metadata["source"]).relative_to(DATA_DIR).as_posix()
    
    if "page" in document.metadata:
        document.metadata["page"] += 1

# The ready LangChain splitter preserves source/page metadata in every chunk.
chunks = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
).split_documents(documents)

embeddings = create_embeddings()

# Opening Chroma creates the collection. Delete it so every run is a clean rebuild.
store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_configuration={"hnsw": {"space": "cosine"}}
)
store.delete_collection()
store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_configuration={"hnsw": {"space": "cosine"}}
)

# Chroma cannot accept all 13k+ chunks in one call, so we send simple batches.
for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
    batch = chunks[start : start + CHROMA_BATCH_SIZE]
    store.add_documents(batch)
    print(f"Saved {min(start + CHROMA_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

print(f"\nFiles/pages loaded: {len(documents)}")
print(f"Chunks stored:      {len(chunks)}")
print(f"Chroma directory:   {CHROMA_DIR.resolve()}")
