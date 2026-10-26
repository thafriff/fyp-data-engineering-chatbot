"""
Central settings for the FYP data engineering chatbot.

Every path, source rule and tunable parameter lives here and nowhere else
(see CLAUDE.md). Notebooks and src/ modules import from this file instead of
hardcoding values. API keys are NOT stored here: only the names of the
environment variables are listed, and the keys themselves live in .env.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # original sources, read-only, Git ignored
CLEANED_DIR = DATA_DIR / "cleaned"      # extracted clean text (output of notebook 01)
CHUNKS_DIR = DATA_DIR / "chunks"        # chunks + metadata JSON (output of notebook 02)
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"  # vector index, Git ignored

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEST_QUESTIONS_FILE = EVALUATION_DIR / "test_questions.json"   # approved retrieval benchmark questions (n=28)
DOCS_DIR = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Knowledge base sources
#
# Page numbers are 1-based PDF page numbers (the page counter of the PDF viewer,
# not the printed page numbers). A "page_ranges" entry of None means "all pages".
#
# Fields:
#   name          human readable source name
#   licence       licence recorded in docs/ (used later as chunk metadata)
#   path          file inside RAW_DIR (relative path)
#   format        pdf | zip_markdown | zip_course
#   extractor     library used for text extraction
#   source_type   default source_type label written to every chunk's metadata
#   notes         why the rules below were chosen
# ---------------------------------------------------------------------------
SOURCES = {
    "S01": {
        "name": "BCcampus - Database Design (2nd edition, Watt)",
        "licence": "CC BY",
        "path": "S01/S01_Watt_DatabaseDesign_2e.pdf",
        "format": "pdf",
        "extractor": "pdfplumber",
        "source_type": "textbook",
        # Chapters only: skips cover, contents and versioning history.
        # Pages 144-150 are Appendix C (SQL lab) and are kept on purpose.
        "page_ranges": [(8, 150)],
        # Running page header: "19 | Chapter 5 Data Modelling" (even pages) or
        # "Chapter 5 Data Modelling | 20" (odd pages). The page number changes on every page.
        "strip_line_patterns": [
            r"^\s*\d+\s*\|\s*Chapter \d+ .+$",
            r"^\s*Chapter \d+ .+\|\s*\d+\s*$",
            # Front matter pages use roman numerals: "viii | About the Book", "Acknowledgements | ix".
            r"^\s*[ivxlcdm]+\s*\|\s*\S.*$",
            r"^\s*\S.*\|\s*[ivxlcdm]+\s*$",
        ],
        "notes": "An EPUB copy of the same book exists in S01; it is ignored to avoid duplicate chunks.",
    },
    "S02": {
        "name": "OpenStax - Principles of Data Science",
        "licence": "CC BY-NC-SA 4.0",
        "path": "S02/S02_OpenStax_PrinciplesOfDataScience_2025.pdf",
        "format": "pdf",
        "extractor": "pdfplumber",
        "source_type": "textbook",
        # Chapter 1 (What are data and data science) and
        # Chapter 2 (Collecting and preparing data: cleaning, databases).
        # Chapters 3-10 (statistics, ML, visualisation...) are out of scope.
        "page_ranges": [(19, 112)],
        # OpenStax running footer (about half of the pages) and running page header:
        # "30 1 • What Are Data and Data Science?" (even pages) or
        # "1.5 • Data Science with Python 31" (odd pages).
        "strip_line_patterns": [
            r"^\s*Access for free at openstax\.org\s*$",
            r"^\s*\d+\s+\d+\s+•\s+.+$",
            r"^\s*\d+(?:\.\d+)?\s+•\s+.+\s+\d+\s*$",
        ],
        "notes": "Chapters 1-2 only. Chapter 2 has most of the database and data-cleaning content.",
    },
    "S03": {
        "name": "University of Helsinki - Big Data Platforms (Heljanko, 2026)",
        "licence": "see docs/ licence proof",
        # Two files exist: the full lectures (318 pp) and a filtered version (138 pp).
        # The filtered file is a subset of the full one, so only ONE is used to avoid
        # duplicate chunks. Change this path to switch to the full lectures.
        "path": "S03/S03_Heljanko_BigDataPlatforms-2026_FILTERED.pdf",
        "format": "pdf",
        # Lecture slides. DECISION MADE: PyMuPDF. pdfplumber merges words together
        # ("Cloudcomputingisamodelforenabling...": about 17 suspicious words per 1000
        # versus about 2 for PyMuPDF; see notebook 01, section 5). The two extractors
        # that were compared are listed below so the notebook can reproduce the comparison.
        "extractor": "pymupdf",
        "extractors_compared": ["pymupdf", "pdfplumber"],
        "source_type": "lecture_slides",
        "page_ranges": None,
        "strip_characters": ["▶"],  # slide bullet marker
        # PyMuPDF prints some slide titles twice in a row; keep only one copy.
        "collapse_repeated_lines": True,
        # Every slide repeats a running header and a "slide/total" counter on its own line.
        "strip_line_patterns": [
            r"^\s*Big Data Platforms \(5 ECTS\)\s*$",   # running header
            r"^\s*\d{1,3}/\d{1,3}\s*$",                   # slide counter such as "12/44"
        ],
        "notes": "Slides have ~490 characters per page; PyMuPDF keeps word spacing intact.",
    },
    "S04": {
        "name": "OpenStax - Introduction to Computer Science",
        "licence": "CC BY-NC-SA 4.0",
        "path": "S04/S04_OpenStax_IntroductionToComputerScience_2024.pdf",
        "format": "pdf",
        "extractor": "pdfplumber",
        "source_type": "textbook",
        # Chapter 8 (Data Management) only. The rest of the 939-page book
        # (hardware, OS, web development, ...) is off-topic for this chatbot.
        "page_ranges": [(369, 438)],
        # OpenStax running footer (about half of the pages) and running page header:
        # "380 8 • Data Management" (even pages) or
        # "8.3 • Relational Database Management Systems 381" (odd pages).
        "strip_line_patterns": [
            r"^\s*Access for free at openstax\.org\s*$",
            r"^\s*\d+\s+\d+\s+•\s+.+$",
            r"^\s*\d+(?:\.\d+)?\s+•\s+.+\s+\d+\s*$",
        ],
        "notes": "Data management chapter only.",
    },
    "S05": {
        "name": "Data Engineering Wiki",
        "licence": "CC0",
        "path": "S05/S05_DataEngineeringWiki_CC0_2026-10-06.zip",
        "format": "zip_markdown",
        "extractor": "markdown",
        "source_type": "wiki",
        "include_extensions": [".md"],
        # Files that are not course content.
        "exclude_filenames": ["README.md", "LICENSE"],
        # Cleaning rules applied to the Markdown text.
        "strip_frontmatter": True,       # the --- ... --- block at the top of each page
        "resolve_wiki_links": True,      # [[Target|Alias]] -> Alias, [[Target]] -> Target
        # Multi-line regexes applied to the whole file text. They remove the wiki footer
        # (GitHub edit/copy links, feedback links) from its start to the end of the file.
        # Most files mark it with a "%% wiki footer ... %%" line; a few (for example
        # Concepts/Data Processing/Message Broker.md) only have the "## This note in GitHub" heading.
        "strip_text_patterns": [
            r"(?ms)^%% wiki footer: Please don't edit anything below this line %%.*\Z",
            r"(?ms)^## This note in GitHub\s*$.*\Z",
        ],
        "notes": "76 small Markdown files, already well structured.",
    },
    "S06": {
        "name": "DataTalksClub - Data Engineering Zoomcamp",
        "licence": "supervisor approved (see docs/ access record)",
        "path": "S06/S06_DataTalksClub_DEZoomcamp_2026-10-01.zip",
        "format": "zip_course",
        "extractor": "markdown",
        "source_type": "course_notes",   # default for .md files
        # Only the seven module folders are kept. Everything else in the repo
        # (cohorts/, .tmp/, .github/, projects/, scripts/, images) is skipped.
        "include_folders": [
            "01-docker-terraform",
            "02-workflow-orchestration",
            "03-data-warehouse",
            "04-analytics-engineering",
            "05-data-platforms",
            "06-batch",
            "07-streaming",
        ],
        "exclude_folders": ["cohorts", ".tmp", ".github", "projects", "scripts"],
        # Path prefixes (relative to the ZIP's top folder) to skip.
        # - 07-streaming/extras/ : duplicate/example material with few comments.
        # - 07-streaming/code/live/ : a snapshot of the live workshop. Its files are exact copies
        #   (producer_realtime.py, models.py) or older variants (aggregation_job.py,
        #   pass_through_job.py) of the files in 07-streaming/code/src/, which is kept as the
        #   single maintained copy. Its only other files are a 22-character README and a
        #   102-character main.py.
        "exclude_paths": ["07-streaming/extras/", "07-streaming/code/live/"],
        "include_extensions": [".md", ".sql", ".py"],
        # Code files are labelled so retrieval and evaluation can tell them apart.
        "source_type_by_extension": {
            ".md": "course_notes",
            ".sql": "code",
            ".py": "code",
        },
        "exclude_extensions": [".jpg", ".jpeg", ".png", ".svg", ".gif"],
        "strip_frontmatter": True,
        "notes": "07-streaming/extras/ and 07-streaming/code/live/ are excluded (duplicates); 07-streaming/code/src/ is kept.",
    },
    "S07": {
        "name": "Luna Reyes et al. - Data Analytics for Public Policy, section 1.4",
        "licence": "CC BY-NC-SA 4.0",
        "path": "S07/S07_LunaReyesEtAl_DataAnalyticsPublicPolicy_Sec1-4.pdf",
        "format": "pdf",
        "extractor": "pdfplumber",
        "source_type": "textbook",
        "page_ranges": None,
        # The PDF is a browser printout, so web-page chrome must be removed.
        # Whole-text pattern (may span lines): the Pressbooks navigation / promo block.
        "strip_text_patterns": [
            r"(?s)Home Read Sign.*?supports open publishing practices\.[ \t]*\n?",
        ],
        # Line patterns (one line each): the browser print header with date and time
        # ("10/6/26, 2:20 PM 1.4 Data Quality ...") and the footer URL with its page counter
        # ("https://pressbooks.pub/... 1/9").
        "strip_line_patterns": [
            r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\b.*$",
            r"^\s*https://pressbooks\.pub/\S+\s+\d+/\d+\s*$",
            # Pressbooks platform lines at the end of the last page. The copyright / CC BY-NC-SA
            # attribution sentence just above them is kept on purpose (it is the book's licence text).
            r"^\s*Powered by Pressbooks\s*$",
            r"^\s*Pressbooks User Guide \| Pressbooks Directory \| Contact\s*$",
        ],
        "notes": "Short (9 pages) with 10 tables; table handling should be checked.",
    },
}

# Raw files that must never be processed (kept for reference only).
IGNORED_RAW_FILES = [
    "S01/S01_Watt_DatabaseDesign_2e.epub",                      # duplicate of the S01 PDF
    "S03/S03_Heljanko_BigDataPlatforms-2026_Lectures01-09.pdf",  # superset of the filtered S03 file
]

# ---------------------------------------------------------------------------
# Knowledge base topics (the eight topics from the proposal, Section 1.5)
# Used in notebook 02 to label each chunk with a topic (see TOPIC_DESCRIPTIONS).
# ---------------------------------------------------------------------------
TOPICS = [
    "SQL",
    "ETL/ELT",
    "Data Pipelines",
    "Data Cleaning",
    "Data Processing",
    "Data Quality",
    "Data Warehousing",
    "Big Data",
]

# ---------------------------------------------------------------------------
# Text cleaning (applied to all PDF sources after extraction)
# ---------------------------------------------------------------------------
MIN_PAGE_CHARACTERS = 50   # pages with fewer characters are treated as empty and skipped

# ---------------------------------------------------------------------------
# Chunking (PROVISIONAL: tune in notebook 02 after looking at real chunks)
#
# Strategy: split by headings/sections first. Only a section that is still longer than
# CHUNK_SIZE is split again with fixed-size chunks (CHUNK_SIZE / CHUNK_OVERLAP) that
# prefer paragraph, then sentence, then word boundaries. Sections shorter than
# MIN_CHUNK_CHARACTERS are merged into the next section of the same file.
# Finally, neighbouring prose chunks that belong to the SAME section (same file, same
# heading) are merged while the result stays within CHUNK_SIZE. Chunks of different
# sections are never merged just to reach a size target.
# Code files (.py, .sql) are chunked separately by structure, never by paragraph.
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800           # maximum characters per prose chunk
CHUNK_OVERLAP = 100        # characters shared between neighbouring chunks of one long section
MIN_CHUNK_CHARACTERS = 150  # smaller sections are merged into a neighbour
DROP_CHUNKS_BELOW = 50      # a finished chunk shorter than this has no retrievable content (e.g. an index page) and is dropped
MARKDOWN_HEADING_MAX_LEVEL = 3   # '#', '##', '###' start a new section; deeper headings stay inside it
TEXT_SPLIT_SEPARATORS = ["\n\n", ".\n", ". ", "\n", " ", ""]   # preferred split points, best first

CODE_CHUNK_SIZE = 1500     # maximum characters per code chunk (code is split on definitions / statements)

# PDF textbooks have no Markdown headings, so section headings are found with these
# regexes (each must match a whole line). Sources not listed here are handled as follows:
# S03 = one slide per chunk (a slide title is its section), S05/S06 = Markdown headings.
PDF_HEADING_PATTERNS = {
    "S01": [r"^Chapter \d+ [A-Z].{2,90}$"],                    # "Chapter 5 Data Modelling"
    "S02": [r"^\d{1,2}\.\d{1,2} [A-Z][^.]{2,80}$"],           # "1.2 Data Science in Practice"
    "S04": [r"^\d{1,2}\.\d{1,2} [A-Z][^.]{2,80}$"],           # "8.3 Relational Database Management Systems"
    "S07": [r"^\d{1,2}\.\d{1,2} [A-Z][^.]{2,80}$"],           # "1.4 Data Quality and Data Types"
}

# Difficulty label per source, taken from the level of the material (a heuristic, to be
# reviewed): textbooks for beginners = beginner, university lectures/practical courses = intermediate/advanced.
DIFFICULTY_BY_SOURCE = {
    "S01": "intermediate",
    "S02": "beginner",
    "S03": "advanced",
    "S04": "beginner",
    "S05": "intermediate",
    "S06": "intermediate",
    "S07": "beginner",
}

CHUNKS_FILE = CHUNKS_DIR / "chunks.json"     # output of notebook 02

# Metadata fields written for every chunk (see proposal Table 3.10)
CHUNK_METADATA_FIELDS = [
    "chunk_id",
    "source_id",
    "source_name",
    "licence",
    "source_type",   # textbook | lecture_slides | wiki | course_notes | code
    "section",       # nearest heading (chapter / section / slide title / function name)
    "page",          # first PDF page of the chunk (None for Markdown and code)
    "file",          # cleaned text file for PDFs; original relative path for Markdown and code
    "topic",         # one of TOPICS, or TOPIC_OTHER_LABEL (embedding similarity, see below)
    "topic_score",   # cosine similarity between the chunk and its topic description (0-1)
    "keywords",      # KEYWORDS_PER_CHUNK terms with the highest TF-IDF score in the chunk
    "difficulty",
]

# ---------------------------------------------------------------------------
# Topic and keyword labelling (notebook 02)
#
# Topic: each chunk ("section + text") and each topic description below are embedded with
# EMBEDDING_MODEL; the chunk gets the most similar topic. If even the best similarity is below
# TOPIC_MIN_SIMILARITY the chunk is labelled TOPIC_OTHER_LABEL (e.g. Docker setup notes).
# Relational database design (ER modelling, normalization, keys) is described under "SQL",
# because the proposal's eight topics have no separate database-design topic.
# Keywords: TF-IDF over all chunks; the highest-scoring terms of each chunk are kept. A two-word
# phrase is only formed from two words that stand next to each other inside ONE sentence/clause
# and are both not stop words, so phrases never stitch fragments across a sentence or clause break.
# ---------------------------------------------------------------------------
TOPIC_DESCRIPTIONS = {
    "SQL": "SQL and relational databases: SELECT queries, joins, GROUP BY and aggregation, subqueries, "
           "creating and altering tables, primary and foreign keys, constraints, relational database design, "
           "entity relationship modelling and normalization",
    "ETL/ELT": "ETL and ELT processes: extracting data from source systems, transforming it and loading it into a "
               "data warehouse or data lake, data ingestion and data integration",
    "Data Pipelines": "Data pipelines and workflow orchestration: automating and scheduling data flows, DAGs, "
                      "orchestration tools such as Airflow or Kestra, running pipelines in Docker containers, "
                      "infrastructure as code with Terraform",
    "Data Cleaning": "Data cleaning and preprocessing: handling missing values, duplicates, outliers and "
                     "inconsistent formats, preparing and transforming raw data before analysis",
    "Data Processing": "Data processing: batch processing and stream processing, distributed computation with "
                       "Spark, MapReduce and Flink, Kafka message streams, processing large datasets in parallel",
    "Data Quality": "Data quality: accuracy, completeness, consistency, validity and timeliness of data, data "
                    "validation and testing, data governance, metadata and data types",
    "Data Warehousing": "Data warehousing and analytics engineering: data warehouses and data lakes, dimensional "
                        "modelling, star schema, fact and dimension tables, BigQuery, dbt models, business intelligence",
    "Big Data": "Big data platforms: volume, velocity and variety, distributed storage and file systems such as "
                "HDFS, cloud computing, data centres, NoSQL databases, scaling across clusters of machines",
}
TOPIC_MIN_SIMILARITY = 0.25      # below this the chunk is labelled TOPIC_OTHER_LABEL
                                 # (raised from 0.20: chunks scoring 0.20-0.25 were mostly generic
                                 # Python/pandas/statistics material force-fitted into a topic)
TOPIC_OTHER_LABEL = "Other"

KEYWORDS_PER_CHUNK = 5
KEYWORD_NGRAM_RANGE = (1, 2)     # single words and two-word phrases
KEYWORD_MIN_DF = 2               # a term must appear in at least 2 chunks
KEYWORD_MAX_DF = 0.30            # a term in more than 30% of chunks is too common to describe one
KEYWORD_EXTRA_STOP_WORDS = [
    # web / markup leftovers
    "https", "http", "www", "com", "org", "html", "png", "jpg", "svg", "gif", "img", "src", "alt",
    "div", "span", "br", "td", "tr", "stroke", "youtu", "youtube", "watch", "github", "md",
    # pieces of contractions ("we'll" -> "ll")
    "ll", "ve", "re", "don", "didn", "doesn", "isn", "aren", "won", "wasn",
    # generic words that describe no topic
    "figure", "table", "example", "use", "used", "using", "like", "just", "let", "need",
]

# ---------------------------------------------------------------------------
# Embedding and vector store
# ---------------------------------------------------------------------------
# DECIDED in notebook 03 (benchmark on the 100 approved test questions): bge-small-en-v1.5,
# used WITHOUT a query prefix. Against all-MiniLM-L6-v2: hit@1 0.830 vs 0.710, MRR 0.892 vs 0.815,
# hit@5 tied at 0.970, head to head 19 vs 7 (p = 0.03); truncates 3 chunks instead of 116.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_QUERY_PREFIX_IN_USE = ""      # no instruction in front of questions (it scored slightly worse)

# Topic labelling in notebook 02 stays on the model it was calibrated with (TOPIC_MIN_SIMILARITY = 0.25
# was tuned on its scores), so re-running notebook 02 reproduces the approved labels.
TOPIC_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Saved search index (written by notebook 03, read by src/core/retriever.py; Git ignored)
FAISS_INDEX_NAME = "course_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / f"{FAISS_INDEX_NAME}.faiss"
FAISS_ID_MAP_FILE = FAISS_INDEX_DIR / f"{FAISS_INDEX_NAME}_chunk_ids.json"   # row i of the index = chunk id i

# Benchmark (notebook 03): the models that were compared
EMBEDDING_CANDIDATES = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
# What is embedded for each chunk: its section label plus its text (same as the topic labelling).
EMBED_TEXT_TEMPLATE = "{section}\n{text}"
# Some models are trained to expect an instruction in front of a search QUERY (never in front of
# the chunks). bge-small-en-v1.5 recommends this one for short retrieval queries; the notebook
# also measures bge without it, so the effect of the prefix is visible.
EMBEDDING_QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
EMBEDDING_BATCH_SIZE = 64
BENCHMARK_TOP_K = [1, 3, 5, 10]   # hit@k cut-offs; MRR is computed over the top 10

# ---------------------------------------------------------------------------
# Retrieval (PROVISIONAL: tune on the evaluation questions)
# ---------------------------------------------------------------------------
TOP_K = 5
# Below this top-1 cosine score the chatbot gives the fallback answer instead of calling the LLM.
# Provisional value for bge-small-en-v1.5: on the 100 approved questions the lowest top-1 score was
# 0.764; on 15 clearly off-topic questions the highest was 0.731 (median 0.551). To be tuned formally later.
SIMILARITY_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# LLM
# The final model is chosen in notebook 04. Only environment variable NAMES are
# listed here; the real keys are read from .env (never commit them).
# ---------------------------------------------------------------------------
API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
LLM_PROVIDER = "google"
LLM_MODEL = None   # set after the model comparison in notebook 04
LLM_TEMPERATURE = 0.2

FALLBACK_MESSAGE = (
    "I could not find this in the course materials. "
    "Please check your lecture notes or ask your lecturer."
)

# ---------------------------------------------------------------------------
# Intent routing (src/core/intent.py), checked BEFORE retrieval.
# Rule-based, no LLM: only a message that is ENTIRELY a greeting / small-talk phrase, or that is unclear
# (gibberish, a stray character), skips retrieval and gets a fixed reply. Everything else, including
# off-topic questions, goes through retrieval and the SIMILARITY_THRESHOLD fallback unchanged.
# ---------------------------------------------------------------------------
# Patterns are matched against the WHOLE normalised message (lower case, punctuation removed, single spaces),
# so "hi, what is OLAP?" is NOT small talk and still goes to retrieval.
SMALL_TALK_PATTERNS = {
    "greeting": r"(hi+|hello+|hey+|hiya|helo|yo|greetings|salam|assalamualaikum|good (morning|afternoon|evening|day)"
                r"|how are you( doing)?( today)?|how r u|whats up|sup)( there| chatbot| bot| assistant)?",
    "thanks": r"(thanks?|thank you|thank you so much|thanks a lot|thx|ty|cheers|ok thanks|okay thanks)"
              r"( chatbot| bot| assistant)?",
    "goodbye": r"(bye+|goodbye|see you|see ya|good night)( chatbot| bot| assistant)?",
    "acknowledgement": r"(ok|okay|k|cool)",
}
SMALL_TALK_RESPONSES = {
    "greeting": ("Hello! I'm the course assistant for Data Engineering. What would you like help with today? "
                 "You can ask me about SQL, ETL/ELT, data pipelines, data cleaning, data processing, "
                 "data quality, data warehousing or big data."),
    "thanks": "You're welcome! Is there anything else from the course material you'd like me to explain?",
    "goodbye": "Goodbye, and good luck with your studies! Come back any time you have a question about the course.",
    "acknowledgement": "Great! Let me know if there's anything else from the course material you'd like me to explain.",
}
UNCLEAR_RESPONSE = ("Sorry, I'm not sure what you meant. Could you rephrase your question or add a bit more detail? "
                    "For example: \"What is the difference between ETL and ELT?\"")
# A message is "unclear" if, after normalisation, it has fewer than this many letters/digits in total ...
UNCLEAR_MIN_ALNUM_CHARS = 2
# ... or it is one word that is a keyboard mash: a run along one keyboard row (e.g. "asdf", "qwerty") of at least
# this length, or one character repeated (e.g. "aaaa", "zzzz"). Real short terms like "SQL", "ETL", "dbt" pass.
KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
KEYBOARD_MASH_MIN_LENGTH = 4

# LLM label classifier for emotional / personal statements ("im exhausted", "rough day") that the patterns above
# miss. Runs ONLY for messages that are not small talk, not unclear, and not an obvious question. The model may
# output ONLY one label from INTENT_LABELS; the reply is a canned text chosen by code from EMOTIONAL_RESPONSES,
# never generated. Any other output, an error or a timeout falls back to "REAL_QUESTION" (normal retrieval).
INTENT_CLASSIFIER_ENABLED = True
# A message is an "obvious question" (skips the classifier, goes straight to retrieval) if it ends with "?" or
# its first word is one of these.
QUESTION_START_WORDS = {
    "what", "whats", "how", "why", "when", "where", "which", "who", "whom", "whose", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "should", "would", "will", "may", "explain", "define", "describe",
    "compare", "list", "give", "show", "tell", "help", "difference", "differences", "example", "examples",
}
INTENT_LABELS = ["EMOTIONAL_SICK", "EMOTIONAL_TIRED", "EMOTIONAL_STRESSED", "EMOTIONAL_OTHER", "REAL_QUESTION"]
INTENT_FALLBACK_LABEL = "REAL_QUESTION"
INTENT_CLASSIFIER_PROMPT = (
    "You are a message classifier for a Data Engineering course chatbot. You never answer or reply to the "
    "message. You output exactly one label from this list and nothing else:\n"
    "EMOTIONAL_SICK - the student says they are ill, sick or unwell.\n"
    "EMOTIONAL_TIRED - the student says they are tired, exhausted, sleepy or need a break or rest.\n"
    "EMOTIONAL_STRESSED - the student says they are stressed, anxious, overwhelmed, frustrated or having a bad day.\n"
    "EMOTIONAL_OTHER - any other personal or emotional statement that does not ask for information or help.\n"
    "REAL_QUESTION - the message asks for information, an explanation or help on ANY topic, even if it also "
    "mentions feelings, or it is anything else.\n"
    "Output only the label, in capital letters, with no punctuation, explanation or other words."
)
# Few-shot examples (message, label) shown to the classifier as earlier turns. Deliberately different from the
# test messages used to check the feature.
INTENT_CLASSIFIER_EXAMPLES = [
    ("i have a fever today", "EMOTIONAL_SICK"),
    ("so sleepy rn", "EMOTIONAL_TIRED"),
    ("this assignment is stressing me out", "EMOTIONAL_STRESSED"),
    ("my cat died this morning", "EMOTIONAL_OTHER"),
    ("im tired of my pipeline failing, how do i add retries", "REAL_QUESTION"),
    ("star schema vs snowflake schema", "REAL_QUESTION"),
]
INTENT_CLASSIFIER_MAX_TOKENS = 8         # a label is at most ~6 tokens; stops any free-form text early
INTENT_CLASSIFIER_TIMEOUT_SECONDS = 20    # on timeout -> REAL_QUESTION (retrieval)
EMOTIONAL_RESPONSES = {
    "EMOTIONAL_SICK": "Sorry to hear you're not feeling well. Get well soon! Come back whenever you're ready.",
    "EMOTIONAL_TIRED": "No worries, take your time and get some rest. I'll be here when you're ready to continue.",
    "EMOTIONAL_STRESSED": ("That sounds tough. Take a short break if you need one. When you're ready, I can help "
                           "you work through any course topic step by step."),
    "EMOTIONAL_OTHER": ("Thanks for sharing that with me. Whenever you're ready, I'm here to help with any "
                        "Data Engineering topic from the course."),
}

# Prompt used to generate answers from retrieved chunks. Follows proposal Section 3.9: course-assistant persona,
# simple academic English, knowledge base only, steps for complex questions, "do not answer questions directly"
# (scaffolding guardrail, read with Sections 2.4.2-2.4.3: explain concepts, but give hints/steps/pseudo-code
# instead of finished solutions to tasks), and a polite fallback.
SYSTEM_PROMPT = (
    "You are a friendly teaching assistant for an undergraduate Data Engineering course. "
    "Answer the student's question using ONLY the course material excerpts provided in the context. "
    "Explain in simple academic English suitable for undergraduate students; use short steps for 'how' questions. "
    "Act as a coach, not an answer machine: if the student asks you to solve an exercise or assignment, or to "
    "write a complete query, program or solution for their task, do NOT give the finished answer. Instead explain "
    "the relevant concept from the excerpts, give hints, list the steps they should follow, or show short "
    "pseudo-code, and encourage them to try it themselves. Questions about what a concept is, why it is used or "
    "how two concepts differ should still be explained clearly. "
    "After each statement taken from an excerpt, cite its chunk ID in square brackets, for example [S06-0322]. "
    "Do not use outside knowledge and do not invent chunk IDs. "
    "If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: "
    + FALLBACK_MESSAGE
)
# How each retrieved chunk is shown to the model inside the prompt.
CONTEXT_CHUNK_TEMPLATE = "[{chunk_id}] (source: {source_name}; section: {section})\n{text}"
# Repeated after the question: small models follow rules better when they are restated at the end.
ANSWER_REMINDER = (
    "Answer using only the excerpts above. Put the chunk ID in square brackets after every statement you "
    "take from an excerpt, for example [S06-0322]. If the student asks for a full solution to a task, give hints "
    "and steps instead of the finished answer. If the excerpts do not answer the question, reply only: "
    + FALLBACK_MESSAGE
)

# ---------------------------------------------------------------------------
# DEV PROTOTYPE ONLY (src/app chat interface)
# A small local model served by Ollama, used to see the chatbot working end to end before the formal
# LLM comparison in notebook 04. This is NOT part of the methodology, the model comparison or the
# evaluation criteria, and is not reported as a chosen model.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEV_LLM_MODEL = "llama3.2:3b"     # ~2 GB, fits a 4 GB laptop GPU (GTX 1650) almost entirely
DEV_LLM_NUM_CTX = 4096            # prompt (instructions + 5 chunks + question) is ~1,500-2,000 tokens
DEV_LLM_KEEP_ALIVE = "30m"        # keep the model loaded between questions so answers start quickly
DEV_LLM_TIMEOUT_SECONDS = 300
