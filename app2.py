import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# ----------------------------------------------------
# STREAMLIT PAGE SETUP
# ----------------------------------------------------
st.set_page_config(page_title="Ask your PDF (Offline Qwen)", layout="centered")
st.header("📘 Ask your PDF (Offline Qwen – Quantized)")
st.markdown("Upload a PDF and ask questions **completely offline** using **Qwen Q4 (GGUF)**")

# ----------------------------------------------------
# PDF UPLOAD
# ----------------------------------------------------
pdf = st.file_uploader("📂 Upload your PDF file", type="pdf")

# ----------------------------------------------------
# PDF PROCESSING
# ----------------------------------------------------
def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        return None, None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)
    return text, chunks


# ----------------------------------------------------
# VECTOR STORE (FAISS)
# ----------------------------------------------------
def create_knowledge_base(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ----------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------
if pdf is not None:
    text, chunks = process_pdf(pdf)

    if text is None:
        st.warning("⚠️ No extractable text found (scanned PDF).")
        st.stop()

    st.success(f"✅ PDF processed successfully ({len(chunks)} chunks).")

    knowledge_base = create_knowledge_base(chunks)
    st.success("✅ Vector database created.")

    user_question = st.text_input("💬 Ask a question about your PDF:")

    if user_question:
        with st.spinner("🤖 Qwen is thinking..."):
            try:
                # ----------------------------------------------------
                # QWEN GGUF MODEL (BEST QUANTIZED)
                # ----------------------------------------------------
                llm = LlamaCpp(
                    model_path=r"C:\Users\sakec\Downloads\Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                    temperature=0.2,
                    max_tokens=256,
                    n_ctx=4096,
                    verbose=False
                )

                retriever = knowledge_base.as_retriever(
                    search_kwargs={"k": 3}
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )

                response = qa_chain.run(user_question)

                st.subheader("🧠 Answer")
                st.write(response)

            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("👆 Upload a PDF to begin.")
