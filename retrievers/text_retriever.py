import faiss
import numpy as np

class TextRetriever:
    def __init__(self,dim):
        self.index=faiss.IndexFlatL2(dim)
        self.text_chunks=[]

    def add(self,embeddings,chunks):
        vectors=np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def retrieve(self,query_embeddings,k=3):
        D,I=self.index.search(np.array([query_embeddings]).astype("float32"),k=k)
        return [self.text_chunks[i] for i in I[0]]
    