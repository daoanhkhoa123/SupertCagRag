from alpha import vectordatabase as vectordb
from alpha import llm_chain
import streamlit as st
import pdfplumber


""" GLOBAL VARIABLES """

SAMPLE_PATH = ""
MESSAGES = "messages"
VECTOR_DB = "vector_db"
USE_SAMPLE = "use_sample"
FILE_UPLOAD = "file_upload"
PDF_PAGES = "pdf_pages"


""" MAIN CODES """


def setup_session_state():
    """
    Initialize session state variables for chat history and PDF vector database.
    """

    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = []

    if VECTOR_DB not in st.session_state:
        st.session_state[VECTOR_DB] = None

    if USE_SAMPLE not in st.session_state:
        st.session_state[USE_SAMPLE] = False


def handle_upload_pdf(file_upload):
    if st.session_state[VECTOR_DB] is None:
        with st.spinner("Processing uploaded PDF..."):
            st.session_state[VECTOR_DB] = vectordb.create_vector_db()
            vectordb.add_documents(st.session_state[VECTOR_DB], file_upload)

            st.session_state[FILE_UPLOAD] = file_upload
            with pdfplumber.open(file_upload) as pdf:
                st.session_state[PDF_PAGES] = [page.to_image()
                                               for page in pdf.pages]


def display_pdf_pages():
    if PDF_PAGES in st.session_state and st.session_state[PDF_PAGES]:
        zoom_level = st.slider(
            "Zoom Level",
            min_value=100, max_value=1000,
            value=700, step=50, key="zoom_slider"
        )

        with st.container(height=410, border=True):
            for page_image in st.session_state[PDF_PAGES]:
                st.image(page_image, width=zoom_level)


def handle_chat_input():
    message_container = st.container(height=500, border=True)

    for message in st.session_state[MESSAGES]:
        avatar = "🤖" if message["role"] == "assistant" else "😎"

        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a promt here...", key="chat_input"):
        try:
            st.session_state[MESSAGES].append(
                {"role": "user", "content": prompt})
            with message_container.chat_message("user", avatar="😎"):
                st.markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner(":green[processing...]"):
                    if st.session_state[VECTOR_DB] is not None:
                        response = llm_chain.respond_question(
                            st.session_state[VECTOR_DB], prompt)
                        st.markdown(response)

                    else:
                        st.warning("Please upload a PDF file first")

        except Exception as e:
            st.error(f"Error: {e}")

    elif st.session_state[VECTOR_DB] is None:
        st.warning("Upload a PDF file to begin chat...")


def main():
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    # Initialize session state
    setup_session_state()

    # Layout
    col1, col2 = st.columns([1.5, 2])
    file_upload = col1.file_uploader("Upload a PDF file ↓",
                                     type="pdf",
                                     key="pdf_uploader")

    if file_upload:
        handle_upload_pdf(file_upload)

    display_pdf_pages()

    delete_collection = col1.button(
        "⚠️ Delete collection", type="secondary", key="delete_button")
    if delete_collection:
        vectordb.delete_vector_db(st.session_state[VECTOR_DB])

    handle_chat_input()


if __name__ == "__main__":
    main()
