import os
import ollama
import numpy as np


from embeddings.text.text_ingestion import load__and_chunk
from embeddings.text.text_embedder import embed_text
from retrievers.text_retriever import TextRetriever
from llm.ollama_client import generate_answer

pdf_path=r"C:\Users\user\Desktop\AMIT_Workspace\Multimodel-RAG\data\pdfs\sample.pdf"

if not os.path.exists(path=pdf_path):
    raise FileNotFoundError("Add a pdf file inside data/pdfs/sample.pdf")


#--------------chunking----------------
print("Loading and chunking PDF....")

chunk=load__and_chunk(pdf_path=pdf_path)

print(f"Total chunks created:{len(chunk)}")

#-----------Embed Chunks-----------------

print("Generating embedding for chunks...............")
print(type(chunk))
print(len(chunk[0]))
#print(chunk[:200])

chunk_embeddings=embed_text(chunk)
embedding_dim=len(chunk_embeddings[0])


#--------------Store in Faiss----------
retriever=TextRetriever(dim=embedding_dim)
retriever.add(chunk_embeddings,chunks=chunk)
print("Embeddings stored in FAISS index.")


#-----------Query--------------------

query=input("\n Ask your questions")
print("Embedding query..........")
query_embedding=ollama.embeddings(model="nomic-embed-text",prompt=query)["embedding"]

top_chunks=retriever.retrieve(query_embeddings=query_embedding,k=3)

context="\n\n".join(top_chunks)

print("\nRetrieved Context:")
print("=" * 50)
print(context[:500])
print("=" * 50)


# -----------------------------
# 6️⃣ Generate Answer
# -----------------------------
print("\nGenerating answer using mistral ...")
answer = generate_answer(context, query)

print("\nFinal Answer:")
print("=" * 50)
print(answer)
print("=" * 50)


from retrievers.table_retriever import TableRetriever

table_retriever = TableRetriever()
table_retriever.load_table("data/tables/sample_table_csv.csv")
table_retriever.build_index()

results = table_retriever.retrieve("What was yield in 2022?")
