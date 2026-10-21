"""Minimal Streamlit UI for the local LangChain RAG."""

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from rag_langchain.settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    OLLAMA_MODEL,
    create_embeddings
)


st.set_page_config(page_title="Local RAG", page_icon="📚")
st.title("📚 Local RAG")
st.caption("LangChain + Chroma + SentenceTransformers + Ollama")


@st.cache_resource
def load_rag():
    """Load heavy models and Chroma only once, not after every UI rerun."""

    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=create_embeddings()
    )

    prompt = ChatPromptTemplate.from_template(
        """Ты отвечаешь только по контексту из локальных документов.
Если ответа в контексте нет, скажи: "В документах недостаточно информации".
Номера фрагментов указывай в ответе как [1], [2].

Контекст:
{context}

Вопрос: {question}"""
    )

    chain = prompt | ChatOllama(model=OLLAMA_MODEL, temperature=0.1) | StrOutputParser()
    return store, chain


top_k = st.sidebar.slider("Количество найденных chunks", 1, 10, 5)
question = st.text_input("Вопрос по документам")

if st.button("Спросить", type="primary") and question:
    store, chain = load_rag()

    # Chroma converts the question into an embedding and finds nearest chunks.
    results = store.similarity_search_with_score(question, k=top_k)

    # Only these chunks, not the whole library, go into the LLM prompt.
    context = "\n\n".join(
        f"[{number}] {document.page_content}"
        for number, (document, _) in enumerate(results, start=1)
    )

    answer = chain.invoke(
        {
            "context": context, 
            "question": question
        }
    )

    st.subheader("Ответ")
    st.write(answer)

    # Sources come from Chroma metadata, so the LLM cannot invent them.
    st.subheader("Источники")
    for number, (document, distance) in enumerate(results, start=1):
        page = document.metadata.get("page", "—")
        st.write(f"[{number}] {document.metadata['source']}, page {page}")

        with st.expander(f"Показать chunk {number} · similarity={1 - distance:.3f}"):
            st.write(document.page_content)
