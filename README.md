# Local RAG

Простой полностью локальный RAG по файлам PDF, TXT и Markdown.


- LangChain связывает загрузчики, splitter, Chroma, prompt и Ollama;
- SentenceTransformers создаёт embeddings локально на GPU;
- Chroma хранит chunks, embeddings и metadata;
- Ollama запускает локальную LLM;
- Streamlit показывает интерфейс, ответ и источники.


## Файлы проекта

```text
app/main.py                    Streamlit UI и query pipeline
src/rag_langchain/index.py     indexing pipeline
src/rag_langchain/settings.py  все основные настройки
data/                          локальные документы
storage/                       локальный индекс Chroma
```

## Установка

Требуются Python 3.11+, NVIDIA GPU и Ollama.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct
```

`requirements.txt` использует CUDA-сборку PyTorch для RTX 5070.

## 1. Индексация

Положите файлы в любые подпапки внутри `data/`, например:

```text
data/dl/
data/math/
```

Затем запустите:

```powershell
python -m rag_langchain.index
```

Команда всегда полностью пересобирает collection `langchain_demo`. Файлы из
всех подпапок `data/` находятся рекурсивно.

## 2. Интерфейс

```powershell
python -m streamlit run app/main.py
```

Откройте `http://127.0.0.1:8501`, задайте вопрос и выберите `top-k`.

## Как идёт индексация

Запуск начинается в `src/rag_langchain/index.py`:

1. `DirectoryLoader` рекурсивно находит PDF, TXT и MD.
2. `PyMuPDFLoader` извлекает текст отдельно с каждой страницы PDF.
3. `TextLoader` читает TXT и MD.
4. Каждый текст представлен стандартным LangChain `Document`:
   `page_content` содержит текст, `metadata` содержит файл и страницу.
5. `RecursiveCharacterTextSplitter` режет документы на chunks по 1000
   символов с overlap 200 символов.
6. `HuggingFaceEmbeddings` вызывает локальный SentenceTransformer
   `multilingual-e5-base` и получает embedding каждого chunk.
7. `Chroma.add_documents()` сохраняет текст, metadata и embedding.
8. Chunks добавляются пакетами по 1000, потому что Chroma ограничивает размер
   одной операции вставки.

Это выполняется заранее. Embeddings всех документов не пересчитываются при
каждом вопросе.

## Как обрабатывается вопрос

Запуск UI начинается в `app/main.py`:

1. `load_rag()` открывает существующую Chroma collection.
2. `HuggingFaceEmbeddings` превращает вопрос в query embedding.
3. `similarity_search_with_score()` находит `top-k` ближайших chunks.
4. Тексты найденных chunks объединяются в `context`.
5. LangChain-цепочка

   ```python
   prompt | ChatOllama(...) | StrOutputParser()
   ```

   подставляет `context` и вопрос в prompt, вызывает Ollama и получает строку.
6. Streamlit показывает ответ.
7. Источники строятся программно из metadata Chroma, а не придумываются LLM.

## Что делает LangChain

LangChain здесь является слоем готовых компонентов и orchestration:

- `DirectoryLoader`, `PyMuPDFLoader`, `TextLoader` загружают документы;
- `Document` хранит текст и metadata в стандартном формате;
- `RecursiveCharacterTextSplitter` создаёт chunks;
- `HuggingFaceEmbeddings` адаптирует SentenceTransformers к LangChain;
- `Chroma` адаптирует vector store;
- `ChatPromptTemplate` собирает prompt;
- `ChatOllama` обращается к локальному серверу Ollama;
- `StrOutputParser` превращает ответ модели в обычную строку;
- оператор `|` собирает компоненты в LCEL chain.

LangChain даёт единые интерфейсы и связывает эти компоненты.

## Что делает Chroma

Chroma — persistent vector store. В collection для каждого chunk хранятся:

- embedding vector;
- исходный текст;
- metadata: имя файла и страница;
- внутренний индекс nearest-neighbor search.

При вопросе Chroma получает query embedding, ищет
ближайшие document embeddings и возвращает соответствующие chunks. Генерацией
занимается LLM через Ollama.

## Почему у E5 два префикса

`multilingual-e5-base` обучалась различать запрос и документ:

```text
query: вопрос пользователя
passage: текст chunk
```

В `settings.py` это передано готовому `HuggingFaceEmbeddings` через
`query_encode_kwargs` и `encode_kwargs`. 

## Основные настройки

Все значения находятся в `src/rag_langchain/settings.py`:

- embedding model;
- Ollama model;
- Chroma path и collection;
- chunk size и overlap;
- размеры batch.

После изменения embedding-модели, chunk size или overlap нужно снова выполнить
`python -m rag_langchain.index`.
