import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using the Google Generative AI Embedding model for text embedding
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Define the prompt template for answering questions
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    just say, "answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}
    Question:
    {question}

    Answer:
    """
    
    # Using the Gemini 1.5 Flash-8B model for conversational AI
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-8b", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Load the FAISS index and get embeddings for the user query
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Set allow_dangerous_deserialization=True to load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform a similarity search on the stored documents
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain
    chain = get_conversational_chain()
    
    # Get the response from the conversational AI model
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Output the response
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Chat With Multiple PDF")
    st.header("Chat with PDF using Gemini")
    
    # User input for the question
    user_question = st.text_input("Ask a Question from the PDF files")
    
    if user_question:
        user_input(user_question)
    
    # Sidebar for file uploading
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
