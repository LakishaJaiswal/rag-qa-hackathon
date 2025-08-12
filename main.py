from retriever import SemanticRetriever
from generator import AnswerGenerator
import re

def highlight_keywords(text):
    keywords = ["encryption", "malware", "phishing", "data privacy", "security", "cybersecurity", "attack", "protection"]
    for kw in keywords:
        text = re.sub(rf"(?i)({kw})", r"\033[1m\1\033[0m", text)  # ANSI bold
    return text

def main():
    retriever = SemanticRetriever("data")
    generator = AnswerGenerator()

    print("Ask your questions (type 'exit' to quit):")
    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            break

        docs = retriever.retrieve(query, top_k=5)
        answer, sources = generator.generate_answer(query, docs)

        print("\n💬 Final Answer:\n")
        print(highlight_keywords(answer))
        print(f"\n📄 Source(s): {sources}\n")

if __name__ == "__main__":
    main()
