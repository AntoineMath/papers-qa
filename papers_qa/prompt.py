from typing import List, Tuple


PROMPT_TEMPLATE: str = (
    "You are a helpful research assistant.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Answer ONLY using the provided context below. Do NOT use general knowledge.\n"
    "2. If information is not explicitly in the context, you MUST say 'I don't know'.\n"
    "3. Do not make inferences or assumptions beyond what is stated.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (ONLY from context, or say 'I don't know have enought information from your knowledge base'):"
)


def build_prompt(question: str, contexts: List[Tuple[str, str, str]]) -> str:
    ctx = "\n".join(f"[{f} | {s}] {chunk}" for f, s, chunk in contexts)
    return PROMPT_TEMPLATE.format(context=ctx, question=question)


def print_unique_contexts(contexts: List[Tuple[str, str, str]]) -> None:
    print("Context used:")
    seen = set()
    for file_name, section, _ in contexts:
        key = (file_name, section)
        if key not in seen:
            seen.add(key)
            print(f"- {file_name} | {section}")


