import nltk
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
import streamlit as st
import logging
import os
import tempfile
import nltk


""" CONFIG VARIABLES """
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
logger = logging.getLogger(__name__)
COLLECTION_NAME = "your_collection_name"
PERSIST_DIRECTORY = "your_persist_directory"
logger.info("DocumentProcessor initialized")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=7500, chunk_overlap=100)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

""" MAIN CODES """


def create_vector_db() -> Chroma:
    global embeddings
    vector_db = Chroma(collection_name=COLLECTION_NAME,
                       embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)

    return vector_db


def load_uploadfiles(file_upload):
    """ From file upload, returns Unstructured PDF data """
    logger.info(f"Loading the uploaded file: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as file:
        file.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")

    loader = UnstructuredPDFLoader(path)
    return loader.load()


def add_documents(vector_db, file_upload):
    logger.info("Document split into chunks")
    data = load_uploadfiles(file_upload)
    chunks = text_splitter.split_documents(data)

    vector_db.add_texts([chunk.page_content for chunk in chunks])
    logger.info(
        f"Successfully added {len(chunks)} chunks to the vector database")


def delete_vector_db(vector_db):
    """ Delete the vector database and clear related session state. """
    logger.info("Deleting vector DB")
    try:
        vector_db.delete_collection()

        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)

        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        logger.error(f"Error deleting collection: {e}")
