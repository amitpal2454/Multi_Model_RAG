import pandas as pd
import ollama
import numpy as np
import faiss
file_path=r"C:\Users\user\Desktop\AMIT_Workspace\Multimodel-RAG\data\tables\sample_table_csv.csv"
class TableRetriever:
    def __init__(self):
        self.index=None
        self.tables=[]
    def load_table(self,file_path):
        df=pd.read_csv(file_path)
        table_text=df.to_markdown()
        self.tables.append((df,table_text))

    def build_index(self):
        embeddings=[]
        for _,table_text in self.tables:
            emb=ollama.embeddings(model="nomic-embed-text",prompt=table_text)["embedding"]
            embeddings.append(emb)
        dim=len(embeddings[0])
        self.index=faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float16"))

    def retrieve(self,query,k=3):
        query_emb=ollama.embeddings(model="nomic-embed-text",prompt=query)["embedding"]
        D,I=self.index.search(np.array([query_emb]).astype("float16"),k=k)
        return [self.tables[i] for i in I[0]]
