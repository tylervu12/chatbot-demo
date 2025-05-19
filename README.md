# RAG Chatbot Demo (Pinecone, OpenAI, LangSmith, Gradio)

This project implements a Retrieval Augmented Generation (RAG) chatbot featuring:
*   **Backend**: Python with LangChain, OpenAI for embeddings and generation, Pinecone for vector storage.
*   **Frontend**: Interactive web interface using Gradio.
*   **Observability**: Integrated with LangSmith for tracing and logging.
*   **Functionality**: Answers user questions based on a knowledge base stored in Pinecone, displays retrieved sources, and validates answers against these sources.

**Repository:** [https://github.com/tylervu12/chatbot-demo](https://github.com/tylervu12/chatbot-demo)

## Key Features

*   **Dynamic RAG Pipeline**: Retrieves relevant document chunks from Pinecone.
*   **Similarity Score Filtering**: Uses only chunks with a similarity score ≥ 0.5.
*   **LLM-Powered Answers**: Generates responses using OpenAI's `gpt-4o` (configurable).
*   **Answer Validation**: An additional LLM call checks if the generated answer is grounded in the provided context.
*   **Source Display**: Shows titles, similarity scores, and text snippets of the document chunks used.
*   **Interactive UI**: Built with Gradio for easy interaction and demonstration.
*   **LangSmith Integration**: Full tracing of the RAG pipeline for debugging and monitoring.

## Getting Started

### Prerequisites

*   Python 3.9+ (Recommended: Python 3.10+)
*   Access to OpenAI API, Pinecone API, and LangSmith API.
*   A Pinecone index already populated with your pre-chunked data. Each chunk's metadata in Pinecone must include `text`, `section_title`, and `source` fields.

### 1. Clone the Repository

```bash
git clone https://github.com/tylervu12/chatbot-demo.git
cd chatbot-demo
```

### 2. Set Up a Virtual Environment

It's highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of the project. Copy the contents of `.env.example` (if provided, otherwise create from scratch) and fill in your API keys and configuration details:

```env
# OpenAI Configuration
OPENAI_API_KEY="sk-YourOpenAIAPIKey"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small" # Or your preferred embedding model

# Pinecone Configuration
PINECONE_API_KEY="YourPineconeAPIKey"
PINECONE_INDEX_NAME="your-pinecone-index-name"

# LangSmith Configuration (for tracing and observability)
LANGCHAIN_API_KEY="ls__YourLangSmithAPIKey"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="Your LangSmith Project Name (e.g., RAG Chatbot Demo)"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com" # Optional, defaults to this
```

**Important**: If deploying to Hugging Face Spaces, set these environment variables as **Secrets** in your Space settings instead of using a `.env` file.

### 5. Run the Gradio Application Locally

Once your environment is set up and your Pinecone index is ready:

```bash
python app.py
```

This will start a local Gradio server. Open the URL provided in your terminal (usually `http://127.0.0.1:7860`) in your web browser to interact with the chatbot.

## Project Structure

```
chatbot-demo/
├── .env                # Local environment variables (DO NOT COMMIT)
├── app.py              # Gradio application frontend and interaction logic
├── main.py             # Core RAG pipeline logic (embeddings, Pinecone query, LLM calls, validation)
├── requirements.txt    # Python package dependencies
├── README.md           # This file
├── instructions.md     # Original system design instructions (for reference)
└── datasets/           # Optional: For evaluation datasets like test-dataset1.csv
    └── ...
```

## How It Works (Flow Overview)

1.  **User Query**: User inputs a question through the Gradio interface.
2.  **Embedding**: The user query is embedded using an OpenAI model.
3.  **Pinecone Query**: The embedding is used to query a Pinecone index for the top 3 most similar document chunks.
4.  **Filtering**: Chunks with a similarity score below 0.5 are discarded.
5.  **Fallback**: If no valid chunks remain, a fallback message is returned.
6.  **Prompt Construction**: A dynamic prompt is constructed for the LLM, including the user's query and the filtered document chunks.
7.  **Answer Generation**: An OpenAI LLM (e.g., `gpt-4o`) generates an answer based *only* on the provided context.
8.  **Answer Validation**: A second LLM call validates if the generated answer is strictly grounded in the provided document chunks.
9.  **Display**: The answer, sources used (with titles, scores, and text snippets), and validation status are displayed in the Gradio UI.
10. **LangSmith Logging**: All steps, inputs, outputs, and metadata are logged to LangSmith for observability.

## Deployment to Hugging Face Spaces

1.  **Create a Space**: Go to [Hugging Face Spaces](https://huggingface.co/new-space) and create a new Space.
2.  **SDK**: Choose "Gradio" as the SDK.
3.  **Repository**: Connect your GitHub repository (`https://github.com/tylervu12/chatbot-demo`).
4.  **Secrets**: Add your API keys and environment variables (from your `.env` file) as **Secrets** in the Space settings. This is crucial for the app to function.
5.  **Files**: Ensure `app.py` (as `app_file` in README metadata), `main.py`, `requirements.txt`, and this `README.md` are correctly pushed to your repository.
6.  The Space should automatically build and deploy your application.
