# retriever.py

from sentence_transformers import SentenceTransformer, util

class SemanticRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.model.encode(documents, convert_to_tensor=True)

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.doc_embeddings, top_k=top_k)[0]

        results = []
        for hit in hits:
            doc = self.documents[hit['corpus_id']]
            score = hit['score']
            results.append((doc, score))
        return results
