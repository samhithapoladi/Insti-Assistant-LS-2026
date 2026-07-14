# IITB Insti-Assist

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using only the information present in the provided IIT Bombay documents. The system retrieves relevant document sections using semantic search and generates grounded responses with Google's Gemini API.

---

## Features

- Supports PDF, TXT, and Markdown documents
- Automatically extracts text from PDF files
- Splits documents into overlapping chunks for better retrieval
- Generates embeddings using **all-MiniLM-L6-v2**
- Uses **FAISS** for fast semantic similarity search
- Generates answers using **Google Gemini**
- Returns the source document and page number (when available)
- Refuses to answer questions not supported by the indexed documents

---

## Project Structure

```text
IITB-Insti-Assist/
│
├── app.py                
├── main_file.py           
├── requirements.txt
├── README.md
│
└── docs/                  
```

---

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- SentenceTransformers
- FAISS
- PyPDF
- NumPy

---

## How It Works

### 1. Document Loading

The application loads all supported files from the selected document folder.

Supported formats:

- PDF
- TXT
- MD

---

### 2. Text Extraction

- PDF files are processed page by page using **PyPDF**.
- Text files and Markdown files are read directly.
- Source filename and page number are stored for citation.

---

### 3. Document Chunking

Each document is divided into overlapping text chunks.

Current configuration:

- Chunk size: **800 characters**
- Chunk overlap: **150 characters**

This preserves context between adjacent chunks.

---

### 4. Embedding Generation

Each chunk is converted into a dense vector using the SentenceTransformers model:

```
all-MiniLM-L6-v2
```

The embeddings are normalized before indexing.

---

### 5. Vector Index

The embeddings are stored in a **FAISS** index for efficient similarity search.

The index is created automatically when the documents are indexed. The generated files are stored in a `vector_store/` directory, which is created by the application if it does not already exist.

---

### 6. Retrieval

When a user submits a query:

1. The query is embedded.
2. FAISS retrieves the four most relevant chunks.
3. Chunks below the similarity threshold are discarded.

---

### 7. Response Generation

The retrieved chunks are passed to Google Gemini along with instructions that require the model to:

- Answer only from the retrieved information.
- Avoid using outside knowledge.
- Avoid making assumptions.
- Return

```
I don't know based on the available documents.
```

when sufficient information is unavailable.

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd IITB-Insti-Assist
```

Create and activate a virtual environment.


```bash
python -m venv venv
venv\Scripts\activate
```


Install the required libraries.

```bash
pip install streamlit numpy faiss-cpu sentence-transformers pypdf google-genai
```

---

## Setting the Gemini API Key

```bash
setx GEMINI_API_KEY "YOUR_API_KEY"
```

Restart the terminal after setting the environment variable.


## Running the Application

Launch the Streamlit application.

```bash
streamlit run app.py
```

---

## Using the Application

1. Upload documents or select a folder containing your IIT Bombay documents.
2. Build or load the document index.
3. Enter your question.
4. The application retrieves the most relevant document chunks.
5. Google Gemini generates an answer using only the retrieved information.
6. The answer and supporting source document(s) are displayed.

---

## Limitations

- Answers are limited to the indexed documents.
- Scanned PDFs without extractable text are not supported.
- Retrieval quality depends on the quality of the uploaded documents.
- A valid Google Gemini API key is required.
- Only PDF, TXT, and Markdown files are supported.

