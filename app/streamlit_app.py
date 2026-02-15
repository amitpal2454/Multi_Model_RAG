import streamlit as st
import pandas as pd
import ollama


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
st.set_page_config(page_title="Multimodal RAG", layout="wide")

import streamlit as st
import os
import ollama
import pandas as pd
import numpy as np

from retrievers.table_retriever import TableRetriever
from retrievers.text_retriever import TextRetriever
from retrievers.image_retriever import ImageRetriever
from embeddings.text.text_ingestion import load_and_chunk
from embeddings.text.text_embedder import embed_text

# ----------------------------------
# CONFIG
# ----------------------------------
LLM_MODEL = "mistral"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Multimodal RAG", layout="wide")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("⚙️ Multimodal RAG")

mode = st.sidebar.radio(
    "Retrieval Mode",
    ["Hybrid (All)", "Text", "Image", "Table"]
)

st.sidebar.markdown("---")
st.sidebar.write(f"LLM: {LLM_MODEL}")
st.sidebar.write(f"Embeddings: {EMBED_MODEL}")

# ----------------------------------
# LOAD TEXT SYSTEM
# ----------------------------------
@st.cache_resource
def load_text_system():
    pdf_folder = "data/pdfs"
    all_chunks = []
    
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            chunks = load_and_chunk(os.path.join(pdf_folder, file))
            all_chunks.extend(chunks)

    if not all_chunks:
        return None

    embeddings = embed_text(all_chunks)
    dim = len(embeddings[0])

    retriever = TextRetriever(dim)
    retriever.add(embeddings, all_chunks)
    return retriever

# ----------------------------------
# LOAD TABLE SYSTEM
# ----------------------------------
@st.cache_resource
def load_table_system():
    table_folder = "data/tables"
    retriever = TableRetriever()

    if not os.path.exists(table_folder):
        return None

    for file in os.listdir(table_folder):
        if file.endswith(".csv"):
            retriever.load_table(os.path.join(table_folder, file))

    if retriever.tables:
        retriever.build_index()
        return retriever
    return None

# ----------------------------------
# LOAD IMAGE SYSTEM
# ----------------------------------
@st.cache_resource
def load_image_system():
    image_folder = "data/images"
    retriever = ImageRetriever()

    if not os.path.exists(image_folder):
        return None

    for file in os.listdir(image_folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            retriever.add_image(os.path.join(image_folder, file))

    return retriever

# ----------------------------------
# LOAD SYSTEMS
# ----------------------------------
text_retriever = load_text_system()
table_retriever = load_table_system()
image_retriever = load_image_system()

# ----------------------------------
# MAIN UI
# ----------------------------------
st.title("🧠 Multimodal RAG (Local + Open Source)")

query = st.text_input("Ask a question")

if query:

    with st.spinner("Processing..."):

        contexts = []
        sources = []
        scores = []

        # ---------------------------
        # TEXT
        # ---------------------------
        if mode in ["Hybrid (All)", "Text"] and text_retriever:

            query_emb = ollama.embeddings(
                model=EMBED_MODEL,
                prompt=query
            )["embedding"]

            retrieved = text_retriever.retrieve(query_emb, k=2)

            for chunk in retrieved:
                contexts.append(chunk)
                sources.append("PDF Document")
                scores.append(0.5)

        # ---------------------------
        # TABLE
        # ---------------------------
        if mode in ["Hybrid (All)", "Table"] and table_retriever:

            table_results = table_retriever.retrieve(query, k=1)

            if table_results:
                df, markdown = table_results[0]
                contexts.append(markdown)
                sources.append("CSV Table")
                scores.append(0.3)

                with st.expander("📊 Retrieved Table"):
                    st.dataframe(df)

        # ---------------------------
        # IMAGE
        # ---------------------------
        if mode in ["Hybrid (All)", "Image"] and image_retriever:

            image_results = image_retriever.search(query, k=1)

            for img_path in image_results:
                contexts.append(f"Relevant image: {img_path}")
                sources.append(img_path)
                scores.append(0.2)

                with st.expander("🖼 Retrieved Image"):
                    st.image(img_path, use_column_width=True)

        # ---------------------------
        # BUILD CONTEXT
        # ---------------------------
        final_context = "\n\n".join(contexts)

        # ---------------------------
        # GENERATE ANSWER
        # ---------------------------
        prompt = f"""
        Answer strictly using the provided context.
        If answer not found, say "Not found in documents".

        Context:
        {final_context}

        Question:
        {query}
        """

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]

        # ---------------------------
        # CONFIDENCE (Simple Heuristic)
        # ---------------------------
        confidence = min(0.95, sum(scores))

    # ----------------------------------
    # DISPLAY RESULTS
    # ----------------------------------
    st.markdown("## ✅ Final Answer")
    st.write(answer)

    st.metric("Confidence Score", f"{confidence:.2f}")

    with st.expander("📚 Sources Used"):
        for s in sources:
            st.write(s)
