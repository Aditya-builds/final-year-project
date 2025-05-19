import os
import tempfile
import streamlit as st
from streamlit_chat import message
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import docx2txt
import PyPDF2
import tiktoken
import json
import datetime
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Document RAG Chat", page_icon="📚", layout="wide")

# Paths for persistent storage
CHAT_HISTORY_PATH = "chat_history.json"
LOG_FILE_PATH = "chat_logs.txt"

# Initialize session state
if "messages" not in st.session_state:
    # Try to load previous messages if they exist
    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, "r") as f:
                st.session_state.messages = json.load(f)
        except Exception as e:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload some documents, and I'll answer questions about them."}]
    else:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload some documents, and I'll answer questions about them."}]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Document processing helpers
def extract_text_from_pdf(file):
    """Extract text from PDF files"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    
    try:
        pdf_reader = PyPDF2.PdfReader(temp_path)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX files"""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp:
        temp.write(file.getvalue())
        temp_path = temp.name
    
    try:
        text = docx2txt.process(temp_path)
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return text

def extract_text_from_txt(file):
    """Extract text from TXT files"""
    return file.getvalue().decode("utf-8")

def process_documents(files, chunk_size):
    """Process uploaded documents and split into chunks"""
    all_docs = []
    
    with st.spinner("Processing documents..."):
        for file in files:
            try:
                # Process based on file type
                if file.name.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif file.name.lower().endswith('.docx'):
                    text = extract_text_from_docx(file)
                elif file.name.lower().endswith('.txt'):
                    text = extract_text_from_txt(file)
                else:
                    st.error(f"Unsupported file type: {file.name}")
                    continue
                
                # Skip empty documents
                if not text.strip():
                    st.warning(f"Could not extract text from {file.name}")
                    continue
                
                all_docs.append(Document(page_content=text, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
    
    if not all_docs:
        st.error("No valid documents were processed.")
        return []
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    chunks = text_splitter.split_documents(all_docs)
    return chunks

def create_embeddings_and_vectorstore(chunks, api_key):
    """Create embeddings and vector store from chunks"""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

def create_conversation_chain(vector_store, api_key, temperature, answer_style):
    """Create conversation chain for RAG"""
    try:
        # Set max_tokens based on answer style
        if answer_style == "Concise":
            max_tokens = 150  # Shorter responses for concise style
        else:
            max_tokens = 1000  # Longer responses for detailed style
        
        # Create the LLM with max_tokens parameter
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=max_tokens  # Add max_tokens parameter
        )
        
        # Create a prompt template based on answer style
        if answer_style == "Concise":
            template = """You are a helpful AI assistant. Answer questions concisely based on the following context.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep your response brief and to the point, focusing only on the most important information.
            
            Context: {context}
            
            Question: {question}
            Concise answer:"""
        else:
            template = """You are a helpful AI assistant. Answer questions with detailed explanations based on the following context.
            Provide comprehensive information, examples, and elaboration where relevant.
            Include additional context, background information, and multiple perspectives when possible.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            Detailed answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        # Create retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
        
        return chain
    
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def save_chat_history():
    """Save the current chat history to a JSON file"""
    try:
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(st.session_state.messages, f)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def log_conversation(question, answer):
    """Log the question and answer to a text file"""
    try:
        # Create log directory if it doesn't exist
        log_dir = Path(LOG_FILE_PATH).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE_PATH, "a") as f:
            f.write(f"[{timestamp}] User: {question}\n")
            f.write(f"[{timestamp}] Assistant: {answer}\n\n")
    except Exception as e:
        st.error(f"Error logging conversation: {e}")

def clear_chat():
    """Clear the chat history"""
    st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. How can I help you with your documents?"}]
    st.session_state.chat_history = []
    save_chat_history()

# Main application interface
st.title("📚 Document Q&A with RAG")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key, including 'sk-' prefix")
    st.divider()
    
    # Document processing settings
    st.subheader("Document Settings")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100, 
                          help="Size of document chunks (smaller = more precise, larger = more context)")
    
    # Model settings
    st.subheader("Model Settings")
    answer_style = st.radio("Answer Style", ["Concise", "Detailed"], 
                           help="Choose between brief answers or detailed explanations")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                           help="Lower = more factual, Higher = more creative")
    
    # Response length display - show the current token limit based on selection
    st.info(f"Response length: {'Up to 150 tokens' if answer_style == 'Concise' else 'Up to 1000 tokens'}")
    
    # File uploader
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    
    process_btn = st.button("Process Documents")
    
    # Add clear chat button to sidebar
    st.divider()
    if st.button("Clear Chat", key="clear_chat"):
        clear_chat()

# Process documents when button is clicked
if process_btn and uploaded_files and openai_api_key:
    # Process documents
    chunks = process_documents(uploaded_files, chunk_size)
    if chunks:
        st.sidebar.success(f"Successfully processed {len(chunks)} document chunks")
        
        # Create vector store
        vector_store = create_embeddings_and_vectorstore(chunks, openai_api_key)
        if vector_store:
            st.session_state.vector_store = vector_store
            
            # Create conversation chain
            conversation_chain = create_conversation_chain(
                vector_store, 
                openai_api_key, 
                temperature, 
                answer_style
            )
            if conversation_chain:
                st.session_state.conversation_chain = conversation_chain
                st.sidebar.success("Ready to answer questions about your documents!")
            else:
                st.sidebar.error("Failed to create conversation chain")
        else:
            st.sidebar.error("Failed to create vector store")
elif process_btn and not uploaded_files:
    st.sidebar.warning("Please upload documents first")
elif process_btn and not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle input
if prompt := st.chat_input("Ask a question about your documents"):
    # Check if documents have been processed
    if not st.session_state.conversation_chain:
        st.error("Please upload and process documents first")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation_chain(
                        {"question": prompt, "chat_history": st.session_state.chat_history}
                    )
                    answer = response["answer"]
                    
                    # Update chat history
                    st.session_state.chat_history.append((prompt, answer))
                    
                    # Display response
                    st.write(answer)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Save chat history to file
                    save_chat_history()
                    
                    # Log conversation to text file
                    log_conversation(prompt, answer)
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")