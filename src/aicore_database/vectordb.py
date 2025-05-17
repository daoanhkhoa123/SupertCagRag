import os
import shutil
import tempfile
from typing import List

import fitz  # PyMuPDF
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

# Only needed for video processing
from moviepy.editor import VideoFileClip
from transformers import pipeline


def create_vector_db(file_upload, logger, doc_embedder):
    """
    Create a vector database from an uploaded PDF file.
    - Only text is split and embedded.
    - Images/tables are NOT summarized or embedded.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file_upload.name)

    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())
    logger.info(f"File saved to temporary path: {pdf_path}")

    haystack_documents: List[HaystackDocument] = []

    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                haystack_documents.append(
                    HaystackDocument(
                        content=text.strip(),
                        meta={
                            "source": f"{file_upload.name}_page_{page_num + 1}",
                            "type": "text"
                        }
                    )
                )

    # Split only text documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    split_haystack_documents = []
    for doc in haystack_documents:
        chunks = text_splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            split_haystack_documents.append(
                HaystackDocument(
                    content=chunk.strip(),
                    meta={**doc.meta, "chunk_id": i}
                )
            )

    logger.info(
        f"Total processed chunks (text only): {len(split_haystack_documents)}"
    )

    docs_with_embeddings = doc_embedder.run(documents=split_haystack_documents)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_with_embeddings["documents"])

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return document_store


def create_vector_db_from_video(model, embedder, video_path: str, logger) -> InMemoryDocumentStore:
    """
    Extracts audio from a video, transcribes it to English, translates to Vietnamese, and stores in a vector DB.

    Args:
        model: LLM model for translation.
        embedder: Embedding model.
        video_path (str): Path to the MP4 video file.
        logger: Logger object.

    Returns:
        InMemoryDocumentStore: Haystack in-memory vector database with embedded Vietnamese chunks.
    """
    raise NotImplementedError("Not yet implemented")

    # --- The following is a template for future implementation ---

    # logger.info(f"Processing video: {video_path}")

    # # Step 1: Extract audio
    # temp_dir = tempfile.mkdtemp()
    # audio_path = os.path.join(temp_dir, "temp_audio.mp3")

    # video_clip = VideoFileClip(video_path)
    # video_clip.audio.write_audiofile(audio_path, logger=None)
    # video_clip.close()

    # # Step 2: Transcribe with Whisper
    # logger.info("Transcribing to English...")
    # transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    # result = transcriber(audio_path, return_timestamps=True)
    # transcript_text = result["text"]
    # os.remove(audio_path)
    # os.rmdir(temp_dir)

    # # Step 3: Split into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = text_splitter.split_documents([LangChainDocument(page_content=transcript_text)])
    # logger.info(f"Transcript split into {len(chunks)} chunks.")

    # # Step 4: Translate to Vietnamese
    # logger.info("Translating chunks to Vietnamese...")
    # prompt_template = (
    #     "Bạn là một chuyên gia ngôn ngữ học thuật. Hãy dịch nội dung sau sang tiếng Việt với độ chính xác cao nhất, "
    #     "sử dụng văn phong trang trọng, rõ ràng và phù hợp với môi trường giáo dục đại học tại Việt Nam.\n\n"
    #     "Nội dung cần dịch:\n{doc}"
    # )

    # translated_chunks = []
    # for i, chunk in enumerate(chunks):
    #     raw_text = chunk.page_content.strip()
    #     if raw_text:
    #         prompt = prompt_template.format(doc=raw_text)
    #         vn_text = f"Nội dung của video {video_path}+\n" + model.run(prompt)["replies"][0]
    #         translated_chunks.append(HaystackDocument(content=vn_text))
    #         logger.info(f"--- Translated Chunk {i+1} ---\n{vn_text[:300]}...\n")

    # if not translated_chunks:
    #     logger.error("No valid translated chunks found.")
    #     raise ValueError("No translated content to embed.")

    # # Step 5: Embed with Ollama
    # logger.info("Embedding Vietnamese chunks...")
    # document_store = InMemoryDocumentStore()
    # docs_with_embeddings = embedder.run(translated_chunks)
    # document_store.write_documents(docs_with_embeddings["documents"])

    # logger.info("Vector DB created successfully.")
    # return document_store
