from transformers import pipeline
import re

class AnswerGenerator:
    def __init__(self):
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=350,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

    def _clean_response(self, response):
        # Split sentences
        sentences = re.split(r'(?<=[.!?]) +', response.strip())
        seen = set()
        filtered = []
        for i, sent in enumerate(sentences):
            norm = sent.lower().strip()

            # Remove repeated whole sentences
            if norm in seen:
                break

            # Remove immediate phrase repetition (e.g. "Cyber attack is a cyber attack")
            if i > 0:
                prev_norm = sentences[i-1].lower().strip()
                if norm.startswith(prev_norm.split()[0]) and norm.startswith(prev_norm):
                    continue

            filtered.append(sent)
            seen.add(norm)

        return ' '.join(filtered)

    def generate_answer(self, query, docs):
        if not docs:
            return "Sorry, I couldn't find relevant information.", None

        combined_context = "\n\n".join([doc['content'] for doc in docs[:5]])
        if len(combined_context) > 1000:
            combined_context = combined_context[:1000].rsplit('.', 1)[0] + '.'

        source_names = ", ".join([doc['filename'] for doc in docs[:5]])

        prompt = (
            "You are an expert assistant. Use ONLY the context below to answer the question clearly and fully. "
            "Write 3 to 5 complete sentences without repeating phrases or tautologies. "
            "Do NOT add any information not found in the context. Avoid bullet points and repetition. End with a summary sentence.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        response = self.generator(prompt, max_new_tokens=350)[0]['generated_text']
        answer = self._clean_response(response)

        def _starts_with_verb(word):
            verbs = {"is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
                     "can", "will", "shall", "should", "could", "would", "may", "might", "must"}
            return word.lower() in verbs

        question_key = query.split()[0].lower()
        answer_lower = answer.lower()

        if not answer_lower.startswith(question_key):
            if _starts_with_verb(question_key) or question_key in {"what", "why", "how", "when", "where", "who"}:
                answer = f"{query.capitalize()} {answer[0].lower() + answer[1:]}" if answer else f"{query.capitalize()}."
            else:
                answer = f"{query.capitalize()} is {answer[0].lower() + answer[1:]}" if answer else f"{query.capitalize()}."

        if len(answer.split()) < 40:
            prompt2 = (
                "Please elaborate on the following answer in 3 to 5 sentences, "
                "without repetition or bullet points:\n\n"
                f"Answer: {answer}\n\n"
                "Elaborated answer:"
            )
            response2 = self.generator(prompt2, max_new_tokens=350)[0]['generated_text']
            answer = self._clean_response(response2)

            answer_lower = answer.lower()
            if not answer_lower.startswith(question_key):
                if _starts_with_verb(question_key) or question_key in {"what", "why", "how", "when", "where", "who"}:
                    answer = f"{query.capitalize()} {answer[0].lower() + answer[1:]}" if answer else f"{query.capitalize()}."
                else:
                    answer = f"{query.capitalize()} is {answer[0].lower() + answer[1:]}" if answer else f"{query.capitalize()}."

        return answer, source_names
