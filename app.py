
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline



# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Ask your PDF (Offline)", layout="centered")
st.header("📘 Ask your PDF (Offline LLM)")
st.markdown("Upload a PDF and ask questions completely offline!")

# -------------------- PDF Upload --------------------
pdf = st.file_uploader("📂 Upload your PDF file", type="pdf")

def process_pdf(file):
    """Extract text from PDF and split into chunks."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        return None, None

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return text, chunks

def create_knowledge_base(chunks):
    """Create FAISS vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

if pdf is not None:
    text, chunks = process_pdf(pdf)
    if text is None:
        st.warning("⚠️ No extractable text found in this PDF (maybe scanned or image-based).")
        st.stop()
    
    st.success(f"✅ PDF processed! Created {len(chunks)} text chunks.")
    
    knowledge_base = create_knowledge_base(chunks)
    st.success("✅ Knowledge base created successfully!")

    # -------------------- User Question --------------------
    user_question = st.text_input("💬 Ask a question about your PDF:")

    if user_question:
        with st.spinner("🔍 Searching for the answer..."):
            try:
                # -------------------- Offline LLM Setup --------------------
                model_path = r"F:\models\flan-t5-small"  # Local path to your downloaded model
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # CPU
                )

                llm = HuggingFacePipeline(pipeline=pipe)

                # -------------------- Build QA Chain --------------------
                retriever = knowledge_base.as_retriever(search_kwargs={"k": 4})
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )

                response = qa.run(user_question)

                st.subheader("🧩 Answer:")
                st.write(response)

            except Exception as e:
                st.error(f"Error while querying the model: {e}")
else:
    st.info("👆 Please upload a PDF file to start.")
