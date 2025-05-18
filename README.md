# Ask the Docs - RAG Application

A powerful Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask natural language questions about their content. The application uses Mistral AI models for both embeddings and question answering.

## 📋 Features

- Upload and process multiple PDF documents
- Extract and chunk text from documents for efficient retrieval
- Generate vector embeddings using Mistral AI
- Ask natural language questions about document content
- Get accurate, context-specific responses from Mistral Large LLM
- View source references to verify information
- Export conversation history as text files
- Clean and intuitive user interface

## 🛠️ How It Works: RAG Methodology

This application implements a Retrieval-Augmented Generation (RAG) pipeline with the following components:

1. **Document Processing**:
   - PDF text extraction using PyPDF2
   - Text chunking with RecursiveCharacterTextSplitter (1000 token chunks with 200 token overlap)
   - Embedding generation using Mistral AI's embedding model
   - Vector storage in FAISS for efficient similarity search

2. **Question Answering**:
   - User question is embedded using the same model
   - Similar chunks are retrieved from the vector store (top 4 most relevant)
   - Retrieved context is sent to Mistral Large LLM with a specialized prompt
   - Response is generated using only the provided context

3. **User Experience**:
   - Two-column layout with document management in sidebar
   - Chat interface in main content area
   - Source references for transparency
   - Conversation history tracking and export
   - Detailed error handling

## 📥 Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd ask-docs
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Mistral AI API key:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

You can obtain a Mistral AI API key from the [Mistral AI Platform](https://console.mistral.ai/).

## 🐳 Docker Deployment

This application can be run using Docker, which simplifies setup and deployment across different environments.

### Prerequisites

- Docker and Docker Compose installed on your system
- A Mistral AI API key

### Using Docker Compose (Recommended)

1. Create a `.env` file in the project root with your Mistral AI API key:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

2. Build and start the application:

```bash
docker-compose up -d
```

3. Access the application at <http://localhost:8501>

4. To stop the application:

```bash
docker-compose down
```

### Using Docker Directly

1. Build the Docker image:

```bash
docker build -t askdocs .
```

2. Run the container:

```bash
docker run -p 8501:8501 -e MISTRAL_API_KEY=your_api_key_here -v $(pwd)/faiss_index:/app/faiss_index askdocs
```

3. Access the application at <http://localhost:8501>

### Persistent Storage

The Docker setup includes a volume to persist the FAISS index between container restarts. This means you only need to process your documents once, and they'll remain available even if you restart the container.

## 🚀 Usage

1. Start the application:

```bash
streamlit run app.py
```

2. The application will open in your default web browser (typically at <http://localhost:8501>)

3. Upload PDF documents:
   - Use the file uploader in the sidebar
   - Click "Process Documents" to extract text and generate embeddings
   - Wait for processing to complete

4. Ask questions about your documents:
   - Type your question in the text input field
   - View the AI-generated answer based on your documents
   - Toggle "Show Source Documents" to see which parts of the document were used

5. Manage your conversation:
   - Use "Clear Chat History" to start a new conversation
   - Download chat history using the provided link

## 📐 Technical Architecture

The application is built with the following components:

- **Frontend**: Streamlit for the web interface
- **Document Processing**: PyPDF2 for PDF parsing
- **Text Chunking**: LangChain's RecursiveCharacterTextSplitter
- **Embeddings**: Mistral AI's embedding model (mistral-embed)
- **Vector Database**: Facebook AI Similarity Search (FAISS)
- **Question Answering**: Mistral Large LLM via LangChain

## 🔒 Security Notes

- The application uses `allow_dangerous_deserialization=True` when loading the FAISS index, which is safe in this context as the index is created within the application.
- Your Mistral AI API key is loaded from the `.env` file and is not exposed in the UI.
- For deployment, ensure you set up proper environment variable management for your API key.

## 🔧 Customization

You can modify the following parameters in the code:

- `chunk_size` and `chunk_overlap` in the `get_text_chunks` function
- Number of retrieved chunks (`k`) in the `answer_question` function
- Temperature and max tokens in the `get_qa_chain` function
- Prompt template for the LLM in the same function

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain) for the RAG components
- [Mistral AI](https://mistral.ai/) for the AI models
- [Streamlit](https://streamlit.io/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database
