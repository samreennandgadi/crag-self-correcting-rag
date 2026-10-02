
import json
import time
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ---------------- CONFIGURATION ----------------

LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3
MAX_RETRIEVAL_RETRIES = 2
MAX_ANSWER_RETRIES = 2

DATA_FILE = Path("data/knowledge.txt")


# ---------------- INITIALIZE MODELS ----------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

vector_db = chromadb.PersistentClient(path="./chroma_db")

# Rebuild the collection each run so document changes are reflected.
try:
    vector_db.delete_collection("knowledge")
except Exception:
    pass

collection = vector_db.create_collection(
    name="knowledge"
)


# ---------------- DOCUMENT INDEXING ----------------

def load_documents():
    text = DATA_FILE.read_text(encoding="utf-8")

    documents = [
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]

    if not documents:
        raise ValueError("No documents found in data/knowledge.txt")

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[f"doc_{i}" for i in range(len(documents))],
        documents=documents,
        embeddings=embeddings
    )

    print(f"Indexed {len(documents)} documents.")


# ---------------- LLM HELPER ----------------

def ask_llm(prompt, json_mode=False):
    options = {"temperature": 0}

    arguments = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": options,
    }

    if json_mode:
        arguments["format"] = "json"

    response = ollama.chat(**arguments)

    return response["message"]["content"]


def ask_llm_json(prompt):
    result = ask_llm(prompt, json_mode=True)
    return json.loads(result)


# ---------------- RETRIEVAL ----------------

def retrieve(query):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count())
    )

    documents = results.get("documents", [[]])[0]
    return documents


# ---------------- DOCUMENT GRADING ----------------

def grade_documents(question, documents):
    relevant = []

    for document in documents:
        prompt = f"""
You are a document relevance grader.

Original question: {question}

Document:
{document}

Decide whether this document contains information useful
for answering the question.

Return JSON with this structure:
{{
  "relevant": true,
  "reason": "short explanation"
}}

Set relevant to false if the document is unrelated.
Do not mark a document relevant just because it shares
a keyword with the question.
"""

        try:
            grade = ask_llm_json(prompt)

            print(
                "Document grade:",
                "RELEVANT" if grade.get("relevant") is True
                else "IRRELEVANT"
            )

            if grade.get("relevant") is True:
                relevant.append(document)

        except (ValueError, TypeError, KeyError) as error:
            print("Could not parse document grade:", error)

    return relevant


# ---------------- QUERY REWRITING ----------------

def rewrite_query(question, documents, feedback=""):
    prompt = f"""
Rewrite the user's question to improve document retrieval.

Original question: {question}

Previous retrieved documents:
{documents}

Feedback:
{feedback}

Return JSON:
{{
  "rewritten_query": "a clearer search query"
}}

Preserve the user's information need.
Do not invent facts.
"""

    try:
        result = ask_llm_json(prompt)
        rewritten = result.get("rewritten_query", "").strip()

        if rewritten:
            return rewritten

    except (ValueError, TypeError, KeyError):
        pass

    return question


# ---------------- CRAG RETRIEVAL LOOP ----------------

def crag_retrieve(question):
    query = question

    for attempt in range(MAX_RETRIEVAL_RETRIES + 1):
        print(f"\nRetrieval attempt {attempt + 1}")
        print("Query:", query)

        documents = retrieve(query)
        relevant_documents = grade_documents(question, documents)

        if relevant_documents:
            print(
                f"Found {len(relevant_documents)} relevant documents."
            )
            return relevant_documents, query

        if attempt < MAX_RETRIEVAL_RETRIES:
            print("Retrieval quality is poor. Rewriting query.")

            query = rewrite_query(
                question,
                documents,
                feedback="No retrieved document passed relevance grading."
            )

    print("Retrieval retry limit reached.")
    return [], query


# ---------------- ANSWER GENERATION ----------------

def generate_answer(question, documents):
    if not documents:
        return (
            "I could not find relevant information in the "
            "available knowledge base."
        )

    context = "\n\n".join(documents)

    prompt = f"""
Answer the user's question using the context below.

Question:
{question}

Context:
{context}

Rules:
- Prefer facts supported by the context.
- Do not invent details.
- If the context is insufficient, clearly say so.
- Give a clear, direct answer.
"""

    return ask_llm(prompt)


# ---------------- ANSWER VALIDATION ----------------

def validate_answer(question, answer, documents):
    context = "\n\n".join(documents)

    prompt = f"""
You are a strict answer validator.

Question:
{question}

Evidence:
{context}

Generated answer:
{answer}

Evaluate:
1. Grounding: Are the factual claims supported by the evidence?
2. Completeness: Does the answer address the question?
3. Hallucination risk: Are there unsupported or potentially
   fabricated factual claims?
4. Retry: Would another retrieval cycle likely improve the answer?

Return JSON exactly in this shape:
{{
  "grounded": true,
  "complete": true,
  "hallucination_risk": "low",
  "retry_needed": false,
  "feedback": "short explanation"
}}

hallucination_risk must be low, medium, or high.
Set grounded to false when important factual claims lack support.
Set complete to false when the question is not adequately answered.
Use the evidence, not your outside knowledge, to judge grounding.
"""

    result = ask_llm_json(prompt)

    risk = result.get("hallucination_risk", "high")
    if risk not in ("low", "medium", "high"):
        risk = "high"

    # Fail closed if the validator returns missing or malformed fields.
    valid = (
        result.get("grounded") is True
        and result.get("complete") is True
        and risk == "low"
    )

    result["hallucination_risk"] = risk
    result["valid"] = valid

    return result


# ---------------- TASK 2: CRAG ----------------

def run_crag(question):
    documents, final_query = crag_retrieve(question)
    answer = generate_answer(question, documents)

    return {
        "question": question,
        "final_query": final_query,
        "documents": documents,
        "answer": answer,
    }


# ---------------- TASK 3: SELF-CORRECTING RAG ----------------

def run_self_correcting_rag(question):
    current_question = question
    last_answer = None

    for attempt in range(MAX_ANSWER_RETRIES + 1):
        print(f"\nAnswer generation attempt {attempt + 1}")

        documents, final_query = crag_retrieve(current_question)
        answer = generate_answer(current_question, documents)

        if not documents:
            print("No evidence found; returning a safe response.")
            return {
                "question": question,
                "answer": answer,
                "valid": False,
                "reason": "No relevant evidence found.",
                "answer_attempts": attempt + 1,
                "retrieval_query": final_query,
            }

        try:
            validation = validate_answer(
                question,
                answer,
                documents
            )
        except (ValueError, TypeError, KeyError) as error:
            print("Validation failed:", error)
            validation = {
                "valid": False,
                "retry_needed": True,
                "feedback": "Answer validation could not be completed.",
                "hallucination_risk": "high",
            }

        print("Validation:", validation)

        last_answer = answer

        if validation.get("valid") is True:
            print("Answer passed validation.")

            return {
                "question": question,
                "answer": answer,
                "valid": True,
                "validation": validation,
                "answer_attempts": attempt + 1,
                "retrieval_query": final_query,
            }

        if attempt < MAX_ANSWER_RETRIES:
            print("Answer failed validation. Retrying retrieval.")

            current_question = rewrite_query(
                question,
                documents,
                feedback=validation.get(
                    "feedback",
                    "Improve the evidence for the answer."
                )
            )

    return {
        "question": question,
        "answer": (
            "I could not produce an answer that passed validation "
            "within the retry limit. Last draft: " + str(last_answer)
        ),
        "valid": False,
        "reason": "Maximum answer retries reached.",
        "answer_attempts": MAX_ANSWER_RETRIES + 1,
    }


# ---------------- INTERACTIVE APPLICATION ----------------

def main():
    load_documents()

    print("\nCRAG + Self-Correcting RAG")
    print("1. CRAG only")
    print("2. Self-correcting RAG")
    mode = input("Choose mode (1 or 2): ").strip()

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    start = time.time()

    if mode == "1":
        result = run_crag(question)
    elif mode == "2":
        result = run_self_correcting_rag(question)
    else:
        print("Invalid mode. Choose 1 or 2.")
        return

    print("\n========== FINAL RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nElapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()