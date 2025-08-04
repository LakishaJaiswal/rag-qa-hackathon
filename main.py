from generator import AnswerGenerator

# Example context documents (you can replace this with your own retriever)
context_docs = [
    ("Malware is software designed to disrupt, damage, or gain unauthorized access to a computer system.", 0.9),
    ("Common types of malware include viruses, worms, trojans, ransomware, and spyware.", 0.85),
    ("Antivirus software helps detect and remove malware from your system.", 0.75)
]

def main():
    generator = AnswerGenerator()

    print("Ask a question:")
    query = input("> ")

    # You would replace `context_docs` with your retriever results
    relevant_docs = context_docs

    final_answer = generator.generate_answer(query, relevant_docs)
    print("\n💬 Final Answer:\n")
    print(final_answer.strip())

if __name__ == "__main__":
    main()
