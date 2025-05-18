import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import time
import base64
from datetime import datetime
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()


mistral_api_key = os.getenv("MISTRAL_API_KEY")

if "api_key" not in st.session_state:
    st.session_state.api_key = mistral_api_key if mistral_api_key else ""


if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = False
if "last_question_sources" not in st.session_state:
    st.session_state.last_question_sources = []


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create vector store with Mistral embeddings"""
  
    if mistral_api_key:
        os.environ["MISTRAL_API_KEY"] = mistral_api_key
    
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_qa_chain():
    """Create QA chain with Mistral LLM"""
   
    
    prompt_template = """
    You are an assistant for question-answering tasks. You will be given context information and a question.
    Your task is to provide a detailed answer based ONLY on the given context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    INSTRUCTIONS:
    1. Answer the question based ONLY on the context provided.
    2. If the answer cannot be determined from the context, say "I cannot answer this question based on the provided context."
    3. Be specific and include relevant details from the context.
    4. Keep your answer concise but complete.
    5. Do not make up or hallucinate information.
    6. Format your answer to enhance readability using markdown when appropriate.
    7. If the answer includes specific facts, numbers, or quotes, cite them directly from the context.
    
    ANSWER:
    """
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,
        max_tokens=1024
    )
    
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    return chain


def format_source(doc):
    """Format a document source for display"""
  
    source_info = ""
    if hasattr(doc.metadata, 'page') and doc.metadata.page is not None:
        source_info = f"Page {doc.metadata.page}"
 
    
    return {
        "content": doc.page_content,
        "source": source_info
    }


def answer_question(user_question):
    """Process user question and generate answer"""
    try:
      
        if mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
        
      
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        
      
        vector_store = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
       
        docs = vector_store.similarity_search(user_question, k=4)
        
      
        st.session_state.last_question_sources = [format_source(doc) for doc in docs]
        
     
        chain = get_qa_chain()
     
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display thinking message
        with st.spinner("Thinking..."):
            # Display a progress message
            placeholder = st.empty()
            placeholder.info("⏳ Searching documents...")
            time.sleep(0.5)
            
            placeholder.info("🔍 Analyzing relevant passages...")
            time.sleep(0.5)
            
            placeholder.info("🤖 Generating response...")
            
          
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            
           
            placeholder.empty()
            
          
            answer = response["output_text"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            return answer
            
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        return error_message


def get_chat_history_text():
    """Generate plain text from chat history"""
    text = "# Ask the Docs - Chat History\n\n"
    text += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if st.session_state.processed_files:
        text += "## Processed Documents\n"
        for file in st.session_state.processed_files:
            text += f"- {file}\n"
        text += "\n"
    
    text += "## Conversation\n\n"
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            text += f"### Question:\n{message['content']}\n\n"
        else:
            text += f"### Answer:\n{message['content']}\n\n"
    
    return text


def get_download_link(text, filename="chat_history.txt", link_text="Download Chat History"):
    """Generate a download link for text file"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def main():
    """Main app function"""
    
    st.set_page_config(
        page_title="Ask the Docs",
        page_icon="📚",
        layout="wide"
    )
    
 
    st.write(css, unsafe_allow_html=True)

   
    col1, col2 = st.columns([1, 3])
    
   
    with col1:
        st.markdown("## 📁 Upload Documents")
        
     
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files to ask questions about"
        )
        
       
        if st.button("Process Documents", type="primary", use_container_width=True):
            if not pdf_docs:
                st.error("Please upload at least one document.")
            elif not mistral_api_key:
                st.error("Mistral API Key is missing from .env file.")
            else:
                with st.spinner("Processing documents..."):
                
                    raw_text = get_pdf_text(pdf_docs)
                
                    text_chunks = get_text_chunks(raw_text)
                    st.info(f"Created {len(text_chunks)} text chunks")
                    
                 
                    get_vector_store(text_chunks)
                    
             
                    st.session_state.processed_files = [pdf.name for pdf in pdf_docs]
                    
               
                    st.success("✅ Documents processed successfully!")
        
      
        if st.session_state.processed_files:
            st.markdown("### Processed Files:")
            for file in st.session_state.processed_files:
                st.markdown(f"- {file}")
    
        st.markdown("### Chat Controls")
        
      
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
        
      
        if len(st.session_state.last_question_sources) > 0:
            st.checkbox("Show Source Documents", key="show_sources")
        
        
        if len(st.session_state.messages) > 0:
            chat_text = get_chat_history_text()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.markdown(
                get_download_link(
                    chat_text, 
                    filename=f"chat_history_{timestamp}.txt", 
                    link_text="📥 Download Chat History"
                ),
                unsafe_allow_html=True
            )
        
        
        st.markdown("---")
        st.markdown("### System Information")
        st.markdown("**LLM:** Mistral Large")
        st.markdown("**Embedding Model:** Mistral Embed")
        st.markdown("**Vector Database:** FAISS")
        
        # API Key status
        if mistral_api_key:
            st.success("✅ Mistral API Key configured")
        else:
            st.error("❌ Mistral API Key missing")
            st.markdown("Add your Mistral API key to the .env file.")
   
    with col2:
        st.markdown("# Ask the Docs 📚")
        st.markdown("### Chat with your documents using Mistral AI")
        
       
        chat_container = st.container()
        with chat_container:
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            
           
            if len(st.session_state.messages) == 0 and not st.session_state.processed_files:
                st.info("👈 Please upload and process documents to start chatting")
        
        
        if st.session_state.show_sources and len(st.session_state.last_question_sources) > 0:
            st.markdown("## Source Documents")
            st.markdown("The following document sections were used to generate the answer:")
            
            for i, source in enumerate(st.session_state.last_question_sources):
                with st.expander(f"Source {i+1} {source['source']}"):
                    st.markdown(source["content"])
        
     
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            if not st.session_state.processed_files:
                st.error("Please upload and process documents before asking questions.")
            elif not mistral_api_key:
                st.error("Mistral API Key is missing from .env file.")
            else:
               
                response = answer_question(user_question)
                
             
                st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)


if __name__ == "__main__":
    main()