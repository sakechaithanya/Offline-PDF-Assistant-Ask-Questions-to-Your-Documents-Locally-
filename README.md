# Offline PDF Assistant: Ask Questions to Your Documents (Locally)
This is a Streamlit application that allows you to chat with your PDF documents completely offline. It uses a local large language model (LLM) and an on-device vector database to answer your questions based on the document's content.

Your data never leaves your machine. All processing, from text extraction to embedding and language model inference, is done locally.

(Note: You should replace this line with a screenshot of your running application!)

# 🚀 Features
100% Offline & Private: Your PDF files and questions are processed locally. No API keys or internet connection are required (after setup).

Chat with any PDF: Upload a PDF and immediately start asking questions.

Local LLM: Uses a local HuggingFacePipeline (e.g., flan-t5-small) to generate answers.

Local Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 to create document embeddings on your device.

Fast & Efficient: Built with Streamlit, LangChain, and FAISS for an efficient in-memory Retrieval-Augmented Generation (RAG) pipeline.

# ⚙️ How it Works
This project implements a local RAG (Retrieval-Augmented Generation) pipeline:

<img width="2048" height="1365" alt="image" src="https://github.com/user-attachments/assets/e27466d5-089a-4096-af6e-bf1e9fb24b22" />


Getty Images

PDF Upload: You upload a PDF file through the Streamlit interface.

Text Extraction: The text is extracted from the PDF using PyPDF2.

Chunking: The extracted text is split into smaller, overlapping chunks using LangChain.

Embedding: Each text chunk is converted into a numerical vector (embedding) using the local SentenceTransformer model.

Indexing (FAISS): The embeddings are stored in a FAISS vector store, which runs entirely in your computer's memory.

Query & Retrieval: When you ask a question:

Your question is also converted into an embedding.

FAISS performs a similarity search to find the most relevant text chunks from the PDF.

Generation (LLM): The question and the retrieved text chunks (the "context") are passed to the local flan-t5-small model, which generates a natural language answer based on the provided context.

Display: The answer is displayed in the Streamlit app.

# 🛠️ Setup & Installation
Follow these steps to get the application running on your local machine.

1. Prerequisites
Python 3.9 or newer

pip (Python package installer)

2. Clone the Repository
Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
3. Create a Virtual Environment (Recommended)
Bash

# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
This project requires several packages. Create a file named requirements.txt and add the following:

requirements.txt

streamlit
PyPDF2
langchain
langchain-community
faiss-cpu
sentence-transformers
transformers
torch
Then, install them all with one command:

Bash

pip install -r requirements.txt
(Note: faiss-cpu is used for simplicity. If you have a compatible GPU, you can install faiss-gpu instead.)

5. Download the Local LLM
This code is configured to load a model from a local directory to ensure it works offline. You must download the model files first.

The code uses google/flan-t5-small.

Go to the Hugging Face model page for flan-t5-small.

Click on the "Files and versions" tab.

Download all the files (especially config.json, pytorch_model.bin, and tokenizer_config.json) and place them in a folder.

For example, create a folder structure like this:

your-project-folder/
|-- models/
|   |-- flan-t5-small/
|       |-- config.json
|       |-- pytorch_model.bin
|       |-- tokenizer.json
|       |-- ... (all other model files)
|-- app.py
|-- requirements.txt
Update the Code: Change the model_path variable in your Python script (app.py) to point to this folder:

Python

# Update this line in your script:
model_path = r"models/flan-t5-small" 
(The script in the prompt used r"F:\models\flan-t5-small", so make sure this path matches wherever you saved your model.)

▶️ How to Run
Once you have installed the dependencies and set up the local model, run the app from your terminal:

Bash

streamlit run app.py
(Rename your script to app.py if you haven't already.)

This will start the Streamlit server, and your default web browser will open to the application's URL (usually http://localhost:8501).

🔧 Customization
Use a different LLM: You can use any Hugging Face model compatible with the AutoModelForSeq2SeqLM class. Just download the model files to a new directory and update the model_path variable. For more powerful generative models (like Llama or Mistral), you would need to change AutoModelForSeq2SeqLM to AutoModelForCausalLM and the pipeline task to "text-generation".

Use a different Embedding Model: Simply change the model_name in the HuggingFaceEmbeddings function call.

Python

# Example:
embeddings = HuggingFaceEmbeddings(model_name="other-sentence-transformer-model")
