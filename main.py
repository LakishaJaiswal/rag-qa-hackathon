# main.py

from retriever import SemanticRetriever
from generator import AnswerGenerator
import os

def load_all_texts(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                # Split each file into docs (double newline as separator)
                docs = f.read().strip().split('\n\n')
                documents.extend(docs)
    return documents

def main():
    # 1. Load all text documents from data folder
    data_folder = 'data'
    documents = load_all_texts(data_folder)

    # 2. Initialize retriever and generator
    retriever = SemanticRetriever(documents)
    generator = AnswerGenerator()

    # 3. Take user input
    print("\n📌 Retrieval-Augmented Q&A System\n")
    query = input("Ask a question:\n> ")

    # 4. Retrieve relevant documents
    relevant_docs = retriever.retrieve(query, top_k=3)

    # 5. Generate the answer
    final_answer = generator.generate_answer(query, relevant_docs)

    # 6. Show results
    print("\n📄 Retrieved Contexts:")
    for i, (doc, score) in enumerate(relevant_docs, 1):
        print(f"\nDoc {i} [Score: {score:.4f}]:\n{doc}")

    print("\n💬 Final Answer:\n", final_answer)

if __name__ == '__main__':
    main()
