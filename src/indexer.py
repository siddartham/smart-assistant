from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .logger import logger

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_and_index(file_paths):
    logger.info(f"Indexing files: {file_paths}")
    docs = []

    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            continue
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local("vectorstore/index")

    logger.info(f"Indexed {len(chunks)} chunks.")
    return f"Indexed {len(chunks)} document chunks."
