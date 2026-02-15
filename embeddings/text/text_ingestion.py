import pdfplumber
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path):
    text=""

    #---------------Better Extraction------------------------------
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted=page.extract_text()
            if extracted:
                text+=extracted+"\n"

    #--------------Embedding model(local)---------------------------
    #embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #model_name="sentence-transformers/all-MiniLM-L6-v2"


    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150,separators=["\n\n","\n","."," ",""])

    #semantic chunkings
    #splitter=SemanticChunker(embeddings)
    chunks=splitter.split_text(text)

    return chunks
