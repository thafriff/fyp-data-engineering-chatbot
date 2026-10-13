# Test questions: known limitations

Applies to `evaluation/test_questions.json` (28 retrieval benchmark questions, approved 2026-10-13),
used in `notebooks/03_embedding.ipynb` to compare embedding models.

## 1. Small sample (n = 28)
28 questions give a **directional signal** between embedding models, not a statistically
bulletproof result. One question is worth about 3.6 percentage points of hit@k, so a few
close-call results flipping between models would not be surprising from noise alone. A
difference of only one or two questions should not be read as a real difference between models.

## 2. Two questions lean on literal identifier matching
- **Q06** ("How is the fct_trips dbt model configured to load data incrementally?")
- **Q11** ("How do you turn on checkpointing in a PyFlink job?")

The exact identifier (`fct_trips`, PyFlink / checkpointing code) appears both in the question and
in the answer chunk, so these two test literal term matching more than pure semantic retrieval.
They stay in the set because students do ask about specific course code, but they should **not
dominate the interpretation** of the results: the notebook also reports the scores without them.

## How relevance is judged
Each question lists every chunk in the knowledge base that answers it (`relevant_chunk_ids`,
100 in total). A retrieval counts as correct if **any** of them appears in the top-k results.
The chunk ids refer to the current `data/chunks/chunks.json` (2,515 chunks); if the chunking
changes, the ids must be re-validated.
