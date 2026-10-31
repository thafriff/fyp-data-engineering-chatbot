# Test questions: known limitations

Applies to `evaluation/test_questions.json`: 100 retrieval benchmark questions, **all approved and frozen**.
Q01-Q28 were approved on 2026-10-13; Q29-Q100 were added at the supervisor's request using the same method and
approved on 2026-10-19. The set is used in `notebooks/03_embedding.ipynb` to compare embedding models.

## 1. Sample size (n = 100)
With 100 questions the benchmark is **considerably more robust** than the original 28:
- one question is now worth 1 percentage point of hit@k (it was about 3.6 points with n = 28);
- the uncertainty of a hit rate around 0.75 is roughly ±8.5 percentage points (95% interval), compared with
  roughly ±16 points at n = 28.

It is still a sample, not a census. Differences of only a few questions between two models remain within
noise, and the head-to-head count (on how many questions one model ranks a correct chunk higher than the
other) is a more sensitive comparison than the difference between two hit rates.

## 2. Questions answered from code identifiers (literal identifier matching)
Ten questions are answered from code identifiers (model, function, command, argument or column names in
code chunks) rather than from prose:

- **Q06** ("How is the fct_trips dbt model configured to load data incrementally?")
- **Q11** ("How do you turn on checkpointing in a PyFlink job?")
- **Q43** ("Which dbt model calculates monthly revenue per pickup zone, and what does it group by?")
- **Q56** ("How does the Spark SQL job script get its input and output paths?")
- **Q58** ("How does the Flink aggregation job count trips per pickup location for each hour?")
- **Q59** ("What does the ROW_NUMBER window function do, and how does PARTITION BY change it?")
- **Q77** ("What do the terraform init, plan and apply commands do?")
- **Q78** ("How does the upload script send a local file to a Google Cloud Storage bucket?")
- **Q79** ("How do you load a large CSV file into Postgres in chunks with pandas?")
- **Q93** ("How does the dbt intermediate trips model remove duplicate trips?")

For these questions, **both retrieval and answer scoring depend on exact identifier matches**. Retrieval
depends on matching names such as `fct_trips`, `enable_checkpointing`, `ROW_NUMBER` or `terraform init`.
Scoring depends on whether an answer reproduces identifiers such as `incremental_strategy='merge'`,
`--input_green` or `if_exists='append'` (see the expected answer points in `evaluation/answer_key.json`).
They test literal term matching more than pure semantic understanding.
In Q06, Q11, Q59 and Q77 the identifier also appears in the question itself. In Q43, Q56, Q58, Q78, Q79 and
Q93 it appears mainly in the answer code. For Q43 the model name `fct_monthly_zone_revenue` appears only in
the chunk's section label, not in its code text.

They stay in the set because students do ask about specific course code and commands, but they should
**not dominate the interpretation** of the results. Note on timing: the notebook 03 sensitivity check (scores
without the literal-identifier questions) excludes only the original four, Q06, Q11, Q59 and Q77; Q06 and Q11
were flagged at n = 28 and Q59 and Q77 in the expansion. The other six were identified later, while drafting
the answer key, so later analyses should report results with and without all ten.

## 3. How relevance was judged
Each question lists every chunk that answers it (`relevant_chunk_ids`). A retrieval counts as correct if
**any** of them appears in the top-k results. Relevant chunks were found by keyword search over the whole
corpus and by reading the hits, **not** with either embedding model, so the labels do not favour one model.
Because one person labelled them by keyword search, a chunk that answers a question in completely different
words may have been missed; such a miss can only make a model's score look lower than it is, and it affects
both models in the same way.

## 4. What the set does not test
- Every question is answerable from the knowledge base; out-of-scope questions (for the fallback answer)
  are not part of this retrieval benchmark.
- Four questions have a single relevant chunk (Q04, Q06, Q15, Q19 in the first 28; Q34, Q43, Q45, Q73 in
  the expansion). They are the strictest test and make individual results noisier.
- Questions were written by the project author from seed chunks, in English; real student phrasing may be
  more varied.

## 5. Composition
Topics are proportional to the number of chunks per topic (Data Warehousing 21, Data Processing 20, SQL 18,
Data Pipelines 13, Data Cleaning 9, Big Data 8, Data Quality 7, ETL/ELT 4). Seed source types: course notes
34, textbook 29, wiki 20, lecture slides 10, code 7.

The chunk ids refer to the current `data/chunks/chunks.json` (2,515 chunks); if the chunking changes, the
ids must be re-validated.
