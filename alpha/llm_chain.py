from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import Chroma
import logging

""" CONFIG VARIABLES """
llm = ChatOllama(model="llama3")
QUERY_PROMPT = PromptTemplate(input_variable=["question"],
                              template="""You are an AI language model assistant. Your task is to generate 2
                                        different versions of the given user question to retrieve relevant documents from
                                        a vector database. By generating multiple perspectives on the user question, your
                                        goal is to help the user overcome some of the limitations of the distance-based
                                        similarity search. Provide these alternative questions separated by newlines.
                      
                                        Original question: {question}""")

template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

logger = logging.getLogger(__name__)

""" MAIN CODES """


def respond_question(vector_db: Chroma, question):
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    logger.info("Processing question and generating response")
    return chain.invoke(question)
