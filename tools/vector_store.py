from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=OllamaEmbeddings(model="jina-embeddings-v2-base-en"),  
    persist_directory="./chroma_langchain_db",
)

def add_documents_to_vector_store(documents):
    vector_store.add_documents(documents=documents)
