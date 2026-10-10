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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Candidate to benchmark against the default in notebook 03 (free, local).
EMBEDDING_MODEL_CANDIDATE = "BAAI/bge-small-en-v1.5"
FAISS_INDEX_NAME = "course_index"

# ---------------------------------------------------------------------------
# Retrieval (PROVISIONAL: tune on the evaluation questions)
# ---------------------------------------------------------------------------
TOP_K = 5
SIMILARITY_THRESHOLD = 0.35   # below this, the chatbot uses the fallback answer

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
