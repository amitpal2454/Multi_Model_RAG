import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor,CLIPModel

image_path=r"C:\Users\user\Desktop\AMIT_Workspace\Multimodel-RAG\data\images\image_4.png"
class ImageRetriever:
    def __init__(self):
        self.model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.proressor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index=faiss.IndexFlatL2(512)
        self.image_paths=[]

    def add_image(self,image_path):
        image=Image.open(image_path)
        inputs=self.proressor(images=image,return_tensors="pt")

        with torch.no_grad():
            embeddings=self.model.get_image_features(**inputs)

        
        vector = embeddings.image_embeds.detach().cpu().numpy().astype("float32")

        self.index.add(vector)
        self.image_paths.append(image_path)

    def search(self,query_text,k=3):
        inputs=self.processor(text=query_text,return_tensors="pt")

        with torch.no_grad():
            text_embeddings=self.model.get_text_features(**inputs)
        
        vector=vector = text_embeddings.image_embeds.detach().cpu().numpy().astype("float32")


        D,I=self.index.search(vector,k)

        return [self.image_paths[i] for i in I[0]]

