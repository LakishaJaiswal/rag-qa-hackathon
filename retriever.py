import os
import glob
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticRetriever:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.documents = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = []
        self.load_documents()

    def load_documents(self):
        for filepath in glob.glob(os.path.join(self.data_folder, "*.txt")):
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            self.documents.append({"filename": filename, "content": text})
            emb = self.model.encode(text, convert_to_tensor=True)
            self.embeddings.append(emb)
        self.embeddings = torch.stack(self.embeddings)

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0].cpu().numpy()
        top_indices = scores.argsort()[::-1][:top_k]
        results = [self.documents[i] for i in top_indices]
        return results
