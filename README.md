# RAG Document Q&A with NVIDIA and OpenAI Embeddings

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for answering questions based on research papers. The application uses LangChain, Streamlit, NVIDIA AI models, and vector embeddings to provide accurate responses to user queries by retrieving relevant information from a collection of PDF documents.

## Features
- **Document Ingestion**: Automatically loads PDF files from a specified directory
- **Text Chunking**: Splits documents into manageable chunks for efficient processing
- **Vector Embeddings**: Creates and stores embeddings using NVIDIA AI or OpenAI technologies
- **Semantic Search**: Finds the most relevant document chunks for a given query
- **AI-Powered Responses**: Generates comprehensive answers using NVIDIA's language models
- **Interactive UI**: Simple Streamlit interface for easy interaction

## Requirements
- Python 3.8+
- Streamlit
- LangChain
- NVIDIA AI Endpoints
- OpenAI API (optional)
- FAISS vector database


```

## Usage

1. Place your PDF research papers in the `./research_papers` directory
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Click the "Document Embeddings" button to process and embed your documents
4. Enter your query in the text input field
5. View the AI-generated answer and explore relevant document sections

## Environment Variables

Create a `.env` file with the following variables:
```
NVIDIA_API_KEY=your_nvidia_api_key
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

## How It Works

1. **Data Ingestion**: The system loads PDF documents from the specified directory
2. **Document Processing**: Documents are split into manageable chunks
3. **Embedding Creation**: Each chunk is converted into a vector embedding
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Query Processing**: User queries are converted to embeddings and similar documents are retrieved
6. **Response Generation**: The LLM generates a response based on the retrieved context

## Customization Options

- Modify chunk size and overlap in the `RecursiveCharacterTextSplitter` settings
- Change the LLM model by updating the `ChatNVIDIA` configuration
- Adjust the prompt template to improve response quality
- Implement alternative embedding models by changing the embeddings class

## Troubleshooting

- Ensure all API keys are correctly set in your environment variables
- Verify that the research papers directory exists and contains PDF files
- Check that model parameters (temperature, top_p, etc.) are appropriate for your use case
- Add debug statements to identify where issues might be occurring

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE)
