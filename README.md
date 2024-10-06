# ChatPDF

ChatPDF is an interactive application that allows users to upload PDF documents and engage in a question-answering session about the content of the document. It uses advanced natural language processing techniques to provide accurate responses and highlight relevant sections in the PDF.

## Features

- PDF document upload
- Question-answering based on the document content
- PDF preview with highlighted excerpts
- Navigation through PDF pages

## Technologies Used

- Streamlit: For the web application interface
- LangChain: For document processing and question-answering
- FAISS: For efficient similarity search and retrieval
- Groq: For the language model
- Ollama: For text embeddings
- PyMuPDF: For PDF processing

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/imanoop7/Highliting-PDF-on-UI-using-Streamlit-for-RAG
   cd chatpdf
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. Ensure Ollama is installed and running with the required model:
   ```bash
   ollama pull nomic-embed-text
   ```

## Running the Application

To run the application, use the following command:

```
streamlit run main.py
```

Then, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Upload a PDF file using the file uploader.
2. Wait for the system to process the document and set up the QA system.
3. Once the system is ready, you can start asking questions about the document content.
4. The application will provide answers and highlight relevant sections in the PDF preview.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
