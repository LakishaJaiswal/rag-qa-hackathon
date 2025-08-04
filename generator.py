from transformers import pipeline

class AnswerGenerator:
    def __init__(self):
        # Using a light model suitable for offline/smaller memory: flan-t5-base
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

    def generate_answer(self, question, docs):
        # Handle both str and (str, score) tuples
        context = "\n\n".join([doc[0] if isinstance(doc, tuple) else doc for doc in docs])

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        result = self.generator(prompt, max_new_tokens=256, do_sample=False)

        return result[0]['generated_text']
