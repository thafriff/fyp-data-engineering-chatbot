# FYP: AI-Powered Chatbot for Data Engineering Students

## Project Overview
Final Year Project (FYP) for Bachelor of Information System (Hons.) 
Intelligent Systems Engineering, UiTM.
Supervisor: Assoc. Prof. Dr. Sofianita Binti Mutalib
Student: Muhammad Althaf Rifqi Mohamed Azlan (2025381467)

## What This Project Does
A RAG-based chatbot that answers Data Engineering course questions
using course materials as the knowledge base. Students ask questions,
the system retrieves relevant chunks from the knowledge base, and
Gemini generates a course-aligned answer.

## Current Status
- Phase 6: Knowledge Base Preparation (IN PROGRESS)
- Data collection DONE: 7 sources collected, licensed, supervisor approved
- Raw files stored locally in data/raw/ (ignored by Git, 161MB)
- Next task: build extraction notebook (01_extraction.ipynb)

## Tech Stack
- Language: Python
- Embedding model: all-MiniLM-L6-v2 (sentence-transformers)
- Vector store: FAISS
- Orchestration: LangChain
- LLM: Gemini API (to be confirmed after model comparison)
- Model comparison: GPT-4o, Gemini, Claude 3.5 Sonnet via API
                    LLaMA, Gemma, Mistral via Groq API (free)
- UI: Streamlit
- Evaluation: RAGAS + expert review

## Folder Structure Rules
- data/raw/        = original source PDFs (Git ignored, kept locally)
- data/cleaned/    = extracted clean text from PDFs
- data/chunks/     = chunked text with metadata JSON files
- data/faiss_index/= stored FAISS vectors (Git ignored)
- src/core/        = RAG logic (rag.py, embedder.py, retriever.py)
- src/app/         = Streamlit UI (app.py)
- config/          = all settings in settings.py only
- notebooks/       = .ipynb files for supervisor documentation
- evaluation/      = test questions and results
- docs/            = licence proofs and source log

## Knowledge Base Sources (7 sources, all open licensed)
- S01: BCcampus
- S02: OpenStax Data Science (CC BY-NC-SA 4.0)
- S03: Helsinki (full 318p + filtered 138p)
- S04: OpenStax Intro to CS (CC BY-NC-SA 4.0)
- S05: DE Wiki (CC0)
- S06: Zoomcamp (supervisor approved)
- S07: Public Policy 1.4 (CC BY-NC-SA 4.0)

## Topics Covered
SQL, ETL/ELT, Data Pipelines, Data Cleaning, Data Processing,
Data Quality, Data Warehousing, Big Data

## Code Rules (IMPORTANT - follow these always)
- One responsibility per file, never mix RAG logic with UI
- All settings go in config/settings.py only, never hardcode
- API keys in .env only, never in code
- Never delete or move anything in data/raw/
- Every notebook must be clean and well commented for supervisor
- Follow separation of concerns (like MVC principle)
- Add comments explaining what each function does

## Notebooks Order (follow this sequence)
1. notebooks/01_extraction.ipynb   - extract text from PDFs
2. notebooks/02_chunking.ipynb     - chunk + label metadata
3. notebooks/03_embedding.ipynb    - embed + store in FAISS
4. notebooks/04_model_comparison.ipynb - compare 6 LLMs
5. notebooks/05_evaluation.ipynb   - RAGAS evaluation

## What NOT to do
- Do not write extraction code yet until told
- Do not modify any file in data/raw/
- Do not push data/raw/ or data/faiss_index/ to GitHub
- Do not mix languages (keep everything in English)
- Do not hardcode any API keys