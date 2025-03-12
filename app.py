import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
import getpass
import os

from dotenv import load_dotenv

load_dotenv()

## Load Groq API

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A"

openapi_key = os.environ["OPENAI_API_KEY"]
lang_chain_key = os.environ["LANGCHAIN_API_KEY"]
nvidia_api_key = os.environ["NVIDIA_API_KEY"]

llm = ChatNVIDIA(
  model="mistralai/mixtral-8x7b-instruct-v0.1",  # Or another appropriate model
  api_key=nvidia_api_key, 
  temperature=0.2,
  top_p=0.9,
  max_tokens=1024,
)



prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI assistant. Answer the following question based only on the provided context.
    
    Context:
    {context}
    
    Question: {input}
    
    Provide a clear, concise, and accurate response based on the information in the context.
    """
)
def create_vector_embeddings():
    print("Starting vector embedding creation")
    if "vectors" not in st.session_state:
        print("Loading documents...")
        st.session_state.loader = PyPDFDirectoryLoader("./research_papers")
        st.session_state.docs = st.session_state.loader.load()
        print(f"Loaded {len(st.session_state.docs)} documents")
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        print(f"Created {len(st.session_state.final_documents)} chunks")
        
        print("Creating embeddings...")
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        print("Vector DB created successfully")

st.title("RAG Document Q&A with NVIDIA and OpenAI Embeddings")

user_prompt = st.text_input("Enter your query from the documents from research papers")

if st.button("Document Embeddings"):
    with st.spinner("Creating vector embeddings... This may take a moment"):
        create_vector_embeddings()
    st.success("Vector DB is created!")


    
import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embeddings' button first to create the vector database.")
    else:
        st.info(f"Processing query: {user_prompt}")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        
        with st.spinner("Searching for relevant documents..."):
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start_time = time.time()  # Use time.time() for wall clock time
            response = retrieval_chain.invoke({"input": user_prompt})
            end_time = time.time()
            
        st.write(f"Response time: {end_time - start_time:.2f} seconds")
        
        if 'answer' in response:
            st.write("### Answer:")
            st.write(response['answer'])
        else:
            st.error("No answer found in the response")
            st.write("Response keys:", list(response.keys()))
