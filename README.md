
# CRAG and Self-Correcting RAG Agent

## 1. Project Overview
Explain corrective retrieval and answer self-validation.

## 2. Architecture
Question -> Retrieve -> Grade -> Rewrite -> Retrieve Again
-> Generate -> Validate -> Final Answer or Retry

## 3. Technology Stack
- Python
- Ollama
- Qwen2.5 3B
- ChromaDB
- Sentence Transformers

## 4. Installation
Document Python setup, dependencies, Ollama installation,
and model download commands.

## 5. How to Run
python main.py

## 6. CRAG Workflow
Explain document relevance grading and bounded retrieval retries.

## 7. Self-Correcting Workflow
Explain grounding, completeness, hallucination risk,
and bounded answer retries.

## 8. Test Cases
Record five questions, actual responses, and observations.

## 9. Failure Handling
Document poor retrieval, validation failures, and retry limits.

## 10. Limitations
LLM grading is imperfect; local knowledge may be incomplete.

## 11. YouTube Demonstration
Add the published video URL.