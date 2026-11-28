# Feature design: study plan, quiz, uploads, read aloud, voice input

**Status: APPROVED by the author on 2026-11-28, with the recommendations:** build order B Quiz me, A Study plan, C Uploads, D Read aloud,
E Voice input (E only if time allows before the freeze); study plan defaults 45-minute sessions, 2 hours a day, limits 20 minutes to 14
days; quiz 2 multiple-choice + 1 short answer, code marks the multiple choice, the student self-marks the short answer against the model
answer; uploads 3 files, 10 MB each, 600-chunk cap, and **C is not built until the author confirms the supervisor has been told**; read
aloud on Gemini Flash-Lite TTS, its price verified (section D). No paid API call was made for this design. Drafted 2026-11-28. Feature A was suggested by the author's lecturer after the 3-minute thesis presentation.

---

## 0. Rules every feature keeps

- **Single-turn chat is unchanged.** A message that triggers none of the new routes, with no upload present, builds byte-identical
  messages to today's. Tests: the 100 evaluation questions and the 10 probes build the same prompts before and after each feature.
- **Notebook 04 is untouched.** The 110 frozen items stay byte-identical, the runners import none of the new modules (tested, as for
  the comparison format), `check_parity.py` passes, and the `SYSTEM_PROMPT` SHA stays 677e8b3b. Every new instruction is app layer,
  appended the way the coach and comparison instructions already are.
- **New routes catch nothing they should not.** None of the 100 evaluation questions, the 10 guardrail probes or the router's held-out
  near-miss set may reach a new route (tested).
- **Models.** Cloud features run on the selected model, `deepseek-flash` (the app default); they work the same with another cloud model,
  but are evaluated on DeepSeek. Local mode uses `qwen3.5:2b`. Read aloud and voice input need Google and are hidden in local mode.
- **Test sets first.** Each feature's test set is written and SHA-256 hashed before that feature's code and never edited afterwards,
  as for the guardrail router and multi-turn sets. Design cases may shape the rules; a separate held-out set (some cases written by
  the author) is reported apart.
- **No paid call in tests.** All tests fake the APIs. Each feature gets one small live check after it is built (costs in section 7).

## Measured on this laptop (2026-11-28, local only)

| Measure | Result |
|---|---|
| Embedding device | **CPU** (the venv's torch has no CUDA; Ollama uses the GTX 1650 separately) |
| bge-small embedding speed | 18-27 chunks per second |
| S01 Database Design (2.8 MB, 153 text pages) | text extraction 0.4 s; 352 chunks; embedding 19 s |
| S03 lecture slides (10.6 MB, 138 slides) | text extraction 0.1 s; 139 chunks; embedding 5 s |
| Search in a separate upload index | 0.3 ms per query |
| Retrieval, "data warehousing", top 40 | 40 at or above 0.70, 13 distinct sections (20 from S04 8.5) |
| Retrieval, "SQL", top 40 | 21 at or above 0.70, 11 sections (mostly S01 chapters 15-16) |
| Retrieval, "ETL and ELT", top 40 | 5 at or above 0.70, 4 sections |
| Retrieval, "machine learning" (outside the course), top 40 | 1 at or above 0.70 (0.704), 1 section |

What these mean for the design: an upload is indexed in seconds, not minutes, but a long text PDF is CPU-bound (about 20 s for 150
pages), so the app needs a progress bar and a chunk cap. Topic labels are noisy (S02 "Data Science with Python" sections are labelled
Data Warehousing, an accepted limitation) and difficulty is set per source, so plans choose sub-topics by retrieval and use the labels
only for ordering. One near-domain chunk at 0.704 shows that the 0.70 threshold alone cannot reject an off-topic plan request.

---

## A) Study plan

**What the student gets.** A time-boxed plan: numbered sessions, each with sub-topics named after real section titles, citations to
the chunks behind them, two or three suggested questions to ask the chatbot, and quiz checkpoints (feature B).

**Triggers and UI.** Both use the same planner function:
- In chat, route `study_plan` (checked after small talk and the task-request check, before the obvious-question check), for messages
  with a planning word and a duration: "I have 3 hours to study data warehousing", "plan 3 days for SQL", "study plan for ETL, 2
  hours". The plan appears as an assistant message.
- A sidebar page **Study plan** with a small form: topic, amount, unit (minutes, hours, days), optional hours per day.

**Parsing (rules, no model call).** A duration pattern (`45 minutes`, `1.5 hours`, `3 days`, `a week`, `half an hour`) and a topic, the
text after "for", "on", "about" or "to study" with planning words removed. A missing duration or topic gets a fixed question back ("How
much time do you have?", "Which topic should the plan cover?"). Days mode assumes **2 hours a day** unless the student says otherwise
("2 hours a day"), and the plan states the assumption.

**Choosing sub-topics (code, from the knowledge base).**
1. Retrieve the top 60 chunks for the topic, through the shared search-text function (greeting removal, spelling correction). If the
   topic names one of the 8 course topics, add up to four expansion queries from that topic's `TOPIC_DESCRIPTIONS`.
2. Keep chunks at or above `SIMILARITY_THRESHOLD` (0.70) and group them by (source, section). Each group is a candidate sub-topic,
   named after its real section title (the last part of an "A > B > C" path, with its parent shown). Groups whose titles normalise to
   the same text are merged and keep all their citations.
3. **In-scope rule:** at least 3 chunks at or above 0.70 from at least 2 sections. This rejects "machine learning" (1 chunk) and accepts
   "ETL and ELT" (5 chunks, 4 sections). These two numbers are fixed on the design cases and then frozen before the held-out run.

**Order.** Difficulty first (the source's beginner, intermediate or advanced label), then course order parsed from the section title
(S01 chapter number, S04 "8.5", S06 module "03-", S05 path order), then retrieval score.

**Time boxing.** Sessions of 45 minutes with a 10-minute break; a final session may be as short as 20 minutes. Minutes go to each
sub-topic in proportion to its number of chunks, with at least 10 minutes each. If the material needs more time than the student has,
the plan keeps the highest-scoring sub-topics, still in order, and lists the rest under "If you have more time". If the course has
less material than the time, the plan adds a review session and an extra quiz, and says how many sections the course has on the
topic. Limits: from 20 minutes to 14 days or 40 hours; outside them the student gets a fixed message (shorter: "start with one
session on X"; longer: "plan the first week").

**What the model writes, and what it does not.** Code builds everything structural: sessions, times, sub-topic names, order and
citations. One model call (a JSON schema) writes, for each session, a one-sentence goal and two or three suggested questions, from
that session's chunks only. **Grounding check:** every suggested question is searched like a student message, and kept only if its
top 5 contains one of the session's own chunks at or above 0.70, which means the chatbot would answer it from the same material. A
failed question is replaced by "What is <sub-topic>?" if that passes, otherwise dropped. Sub-topics cannot be invented, because their
names come from the section titles, not from the model.

**Checkpoints.** After every two sessions (hours mode) or at the end of each day (days mode): "Quiz checkpoint", which runs feature B on
the chunks of the sessions just covered.

**Edge cases.** Off-topic ("plan 3 days for machine learning") is declined with the course's fallback wording and the list of the 8
course topics. Too broad ("all of data engineering in 1 hour") gives an overview plan of the 8 topics, one sentence each, and asks the
student to pick one. A plan request that is also an assessed task ("plan my assignment answer") goes to coach mode, as today.

**Multi-turn.** A plan is not added to the conversation history, so history stays question-and-answer and prompts stay small.
A suggested question the student clicks is sent as a normal chat message.

**Local mode (qwen3.5:2b).** The same structure from code. Goals and questions come from Ollama's structured output (a JSON schema),
with one retry; if both fail, the template questions are used. Expected time 20-40 s, against about 5 s on DeepSeek.

**Output.** A table (session, minutes, sub-topics with numbered citations, suggested questions as buttons), the Sources list used by
chat, quiz-checkpoint buttons and "Download plan (.md)". Nothing is stored.

**Test set and evaluation** (`evaluation/study_plan_cases.json`, hashed): 16 design cases, made up of 8 in scope (one per course topic,
mixing minutes, hours and days, such as "I have 3 hours to study data warehousing", "plan 3 days for SQL", "45 minutes on data quality",
"a week for big data"), 3 off-topic, 2 too broad or too short, 2 missing time or topic, and 1 Malay-English ("plan 2 jam untuk SQL").
A further 6 held-out cases come from the author. Deterministic checks, required in every run:
- the session minutes add up to the budget, within 10%;
- every sub-topic name is a real section title;
- every citation is in the retrieved set and belongs to its session's section;
- difficulty never decreases from one session to the next;
- off-topic requests are declined;
- no plan appears for missing input.

Model checks, 3 runs each on DeepSeek and 1 on qwen3.5:2b:
- at least 90% of suggested questions pass the grounding check before replacement;
- the author reads 5 plans against the source sections (verification against the material, as in the human-review basis).

---

## B) Quiz me

**Triggers and UI.** In chat, route `quiz` for "quiz me on star schema", "test me on SQL joins" or "give me questions on Kafka". It is
also the study plan's checkpoint button, with the session's chunks given directly. The quiz appears as a card inside the assistant
message: three questions in a form, "Check answers", and then each answer, its explanation and its numbered citations are revealed.

**Content.** By default 2 multiple-choice questions (4 options, one correct) and 1 short-answer question, from the top 5 chunks
retrieved for the topic. If the top-1 chunk is below 0.70, the usual fallback is given and no quiz is made.

**Generation.** One model call with a JSON schema. Each item has a type, the question, options for multiple choice, the answer, a
one-sentence explanation, the supporting chunk IDs and, for short answers, 2-3 key points. The app-layer instruction says that every
question must be answerable from the excerpts, every answer must cite them, and nothing may come from outside them.

**Grounding checks (code, before anything is shown):**
- the JSON is valid and has exactly three items;
- every cited ID is in the retrieved set;
- the correct answer is supported by its cited chunk: at least half of the answer's content words appear in that chunk;
- the question, searched like a student message, retrieves its cited chunk in the top 5;
- a multiple-choice item has exactly one correct option, and the options are distinct.

A failed item is generated again once; if it fails again it is dropped and the card says "2 questions" (never invented filler). This
check is lexical and has limits, so the evaluation includes a reading by the author.

**Checking the student's answers.** Multiple choice is checked by code against the stored option. A short answer is **not** graded by
a model: the card reveals the model answer and its key points, and the student ticks "I had this" or "not yet". This avoids grading
errors and keeps the quiz deterministic. An LLM "check my answer" could be added later if wanted.

**Guardrail.** A quiz is practice, not an assessed task. If the request also matches the task-request rule ("quiz me on the answers to
my assignment"), coach mode wins and no quiz is made. **Multi-turn:** quizzes stay out of the history; "quiz me on that" after an
answer uses the follow-up rule to find the earlier topic.

**Local mode.** The same checks, with qwen3.5:2b through Ollama's structured output and one retry. If no item passes, the card says
that a reliable quiz could not be made offline and suggests cloud mode.

**Test set and evaluation** (`evaluation/quiz_cases.json`, hashed): 15 in-scope topics (the 8 course topics plus concepts such as star
schema, normalisation, Kafka topics, dbt tests, OLAP, schema-on-read, SCD), 3 off-topic and 2 assessed-task requests, with 5 held-out
cases. Run 3 times on DeepSeek and once locally. Measures and targets:
- items passing every check on the first try (target 90% or more on DeepSeek);
- items shown after one retry (target 100% of quizzes with at least 2 items);
- no citation outside the retrieved set;
- off-topic requests and tasks routed correctly in every run;
- the author verifies 15 items against their cited chunks (target at least 13 correct).

---

## C) Upload your own materials

**UI.** On the chat page, a sidebar expander **Your materials** with the warning shown above the uploader:
> Only upload material you have the right to use. Do not upload anything with personal information. Uploads are kept only for this
> session and are never saved. In cloud mode, the parts of your file that match a question are sent to DeepSeek to answer it.

`st.file_uploader(type=["pdf", "txt", "md"], max_upload_size=10 MB)`, up to 3 files, each listed with a Remove button.

**Processing (same tools as the knowledge base).**
- PDF with PyMuPDF (the notebook 01 tool) and TXT/MD read as UTF-8; MD is split by headings first (notebook 02's shape), everything
  else at 800 characters with 100 overlap.
- Chunk IDs `U1-0001`, with the file name, page (PDFs) or heading (MD) kept as metadata.
- Embedded with the same, already loaded bge-small into a separate `faiss.IndexFlatIP` held in `st.session_state` only.
- **Never written to disk** (no temporary file: PyMuPDF opens the bytes in memory), never added to `data/` or the frozen index, and
  cleared by **New chat**, by removing the file, and when the app restarts (session state does not survive a restart).

**Rejected with a clear message:**
- a scanned or image-only PDF: fewer than 200 extracted characters in total, or more than half the pages under 50 characters ("This PDF
  has no text layer (it looks scanned). Please upload a text-based PDF; OCR is not supported.");
- an encrypted PDF;
- a file over 10 MB;
- anything above a cap of 600 chunks, about 30 s of embedding on this laptop: the file is indexed up to the cap and the student is told
  which pages were left out.

**Speed on this laptop.** From the measurements: about 1 s for a short notes file, about 5 s for a 10 MB slide deck, and about 20 s for
a 150-page text PDF, with a progress bar. Questions afterwards cost 0.3 ms extra.

**Retrieval with an upload.** The question is embedded once and searched in both indexes; the top 5 by score are kept (the scores are
comparable, being the same model and cosine). The **0.70 threshold and fallback are unchanged** and apply to the best result from
either index. If the upload is relevant it can therefore answer questions the course does not cover. That is intended, and it is
stated in the answer, because excerpts from uploads carry the label "Your upload: <file>, p. <n>".

**Prompt (app layer; SYSTEM_PROMPT unchanged).** Upload excerpts use the same excerpt template, with the source shown as "Student
upload: <file>". When any upload excerpt is retrieved, one app-layer line is appended: excerpts marked "Student upload" come from the
student's own file, not the course materials; cite them by their IDs and say when an answer relies on them. With no upload present, the
messages are byte-identical to today's.

**Citations and display.** `src/core/citations.py` learns the `U<n>-<dddd>` pattern for display only (the notebook 04 citation pattern
stays as it is). In the Sources list an upload citation shows "your file notes.pdf, p. 4".

**Guardrail.** Coach mode is unchanged. Risk: a student uploads an assignment brief and asks "answer question 2 from my file", which may
have no production verb. Planned response: count "my file" / "my upload" / "the uploaded" as an assessed signal when an upload exists.
This is in the test set; if it does not hold, it is reported as a limitation.

**Multi-turn.** Follow-ups search both indexes. History is unchanged (questions plus trimmed answers). New chat clears the upload and the
conversation together.

**Notebooks 04 and 05.** Uploads exist only in the app's session state. The runners and the notebook 05 evaluation never see them, and
uploads are excluded from all scoring.

**Test set and evaluation** (`evaluation/upload_cases.json`, hashed, with fixtures):
- a short notes file in Markdown written for the tests and released by the author as CC0 (committable);
- a few pages of OpenStax *Introduction to Computer Science* outside Chapter 8 (CC BY-NC-SA 4.0, not in the knowledge base), cut from
  the local PDF at test time and never committed;
- an image-only PDF made at test time by rendering a page to an image.

The cases are 6 questions answerable only from the upload (they must cite U-IDs), 4 course questions (S-IDs, the same answers as
without an upload), 3 off-topic (fallback), 2 assignment-style requests about the uploaded file (coach mode), 2 multi-turn follow-ups,
plus the rejection cases (scanned, over 10 MB, encrypted, wrong type).

Checks:
- nothing is written to disk (every file in `data/` and the index keeps its hash, and a temp-folder watch is empty);
- New chat clears the upload;
- with no upload, prompts stay byte-identical;
- indexing time is recorded per fixture;
- every upload citation is labelled with its file and page.

**Report TODOs to record if C is approved** (recorded now as PROPOSED): Section 1.5's "users cannot upload new materials" becomes
"session-only uploads"; Sections 3.9-3.10 describe the design; and an ethics note for the supervisor, since uploads may contain
third-party or personal material, parts go to DeepSeek in cloud mode, and the ERC4 exemption must be re-checked.

---

## D) Read aloud (optional)

**UI.** A "Listen" button under each assistant answer, in cloud mode only (hidden in local mode, and when `GOOGLE_API_KEY` is missing).

**Flow.**
1. Remove citations (`[1]`, `[1, 2]`, `[S06-0204]`), code blocks ("A code example is shown on screen."), table rows, and markdown
   symbols.
2. Send the text to Gemini TTS, `gemini-3.8-flash-lite-tts` (the cheaper of the two on the key's model list), with one fixed voice.
3. The 24 kHz mono WAV is played with `st.audio` and cached in session state, so a replay or rerun is never billed twice.

The answer text goes to Google; the student's own words do not, unless the answer quotes them.

**Cost (VERIFIED 2026-11-28, verbatim from https://aistudio.google.com/docs/pricing, Standard tab, Gemini 3.8 Flash-Lite TTS).** Paid
tier: input "$0.50 (text) through December 31, 2026" ($1.00 from 1 January 2027); output "$6.00 (audio) through December 31, 2026"
($12.00 from 1 January 2027), "Equivalent to $0.0015 per 10s audio"; free tier "Free of charge"; "Audio tokens correspond to 25 tokens
per second of audio." (Flash TTS, for comparison: $0.50 text in, $9.00 audio out.) A 200-word answer is about 80 s of speech, about
2,000 audio tokens: about $0.012 (RM0.05) per listen, plus a negligible text input.

**Tests.** 20 text-cleaning cases (citations, code, tables, markdown, Malay words), the button hidden in local mode, one TTS call per
message (cached), and the cost recorded. One live check: 3 listens, about RM0.2.

---

## E) Voice input (optional, lowest priority)

**UI.** Streamlit 1.65's chat input has a built-in microphone (`st.chat_input(accept_audio=True)`, 16 kHz WAV), shown in cloud mode
only.
1. A recording goes to `gemini-3.8-flash` with a fixed instruction: transcribe exactly, keep Malay words as spoken, keep technical terms
   as written (BigQuery, dbt, OLAP), never translate or answer.
2. The transcript appears in an editable box under the input, with **Send** and **Discard**.
3. Nothing reaches the pipeline until Send, and the audio is never stored.

**Cost.** Audio counts as 32 tokens per second, so a 20 s question is 640 tokens, about $0.0005. **Privacy:** a voice recording is
personal data, so the warning says so and the ethics note covers it.

**Tests.** Faked transcription: the box is prefilled, nothing is sent before Send, and the microphone is hidden in local mode. One live
check: the author records 6 phrases (3 English, 3 Malay-English, such as "apa itu data pipeline", "macam mana nak load data dalam
BigQuery"). Measures: exact technical terms kept, and the transcript needing at most one word changed.

---

## 1. Sidebar and page layout

```
Sidebar                                   Main area (Chat page)
-------------------------------           -----------------------------------------
Pages (st.navigation)                     Data Engineering Course Assistant
  Chat                                    [conversation]
  Study plan                                 assistant answer [1][2]
  Quiz                                       [Listen]                      (D, cloud)
  Model comparison                           Sources (expander)
-------------------------------              quiz card / plan card when asked
Answer model  [deepseek-flash v]
  (status line, provider note)            ---------------------------------------
[ New chat ]                              [ Type your question...   (mic) (E) ]
-------------------------------           Don't type personal information. ... (DeepSeek)
Your materials (expander)  (C)
  warning text
  [ Upload PDF/TXT/MD, 10 MB ]
  notes.pdf (12 chunks)  [Remove]
-------------------------------
Intent classifier: qwen3.5:2b ...
```

The Study plan and Quiz pages are forms around the same functions as the chat commands, so both ways behave identically. Model
comparison stays read-only.

## 2. Recommended build order and estimates

| Order | Feature | Why here | Work (with tests and evaluation) | Live-check cost |
|---|---|---|---|---|
| 1 | B Quiz me | Small, and the study plan's checkpoints depend on it | about 1 day | about RM0.05 (DeepSeek) |
| 2 | A Study plan | The lecturer's request; reuses B | about 1.5 days | about RM0.05 |
| 3 | C Uploads | Largest; changes the report's scope, so it needs the supervisor's OK first | about 2 days | about RM0.05 |
| 4 | D Read aloud | Independent and small | about 0.5 day | about RM0.2 (Google TTS) |
| 5 | E Voice input | Lowest priority; most privacy questions | 0.5-1 day | under RM0.05 |

Each feature is a separate commit: hashed test set first, then the code, the tests, the evaluation and its record.

## 3. Decisions for the author

1. Approve the build order (B, A, C, D, E), or change it.
2. Study plan: confirm 45-minute sessions, a 10-minute break, 2 hours a day by default, and limits of 20 minutes to 14 days or 40 hours.
3. Quiz: confirm 2 multiple-choice and 1 short-answer question, with short answers self-checked (no model grading).
4. Uploads: confirm 3 files, 10 MB each and a 600-chunk cap, and that you will tell your supervisor before C is built.
5. Read aloud: Flash-Lite TTS (cheaper) or Flash TTS; one voice.
6. Whether D and E are built at all (both are marked optional).
