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
        # PROBLEM SOURCE: lecture slides. pdfplumber merges words together
        # ("whichshouldbe,e.g.,atleast2-10xthenumber") and leaves "▶" bullets.
        # Test PyMuPDF first and compare against pdfplumber before choosing.
        "extractor": "pymupdf",
        "fallback_extractor": "pdfplumber",
        "source_type": "lecture_slides",
        "page_ranges": None,
        "problem_source": True,
        "strip_characters": ["▶"],  # slide bullet marker
        "notes": "Slides have ~490 characters per page; check word spacing after extraction.",
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
        "notes": "77 small Markdown files, already well structured.",
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
        "include_extensions": [".md", ".sql", ".py"],
        # Code files are labelled so retrieval and evaluation can tell them apart.
        "source_type_by_extension": {
            ".md": "course_notes",
            ".sql": "code",
            ".py": "code",
        },
        "exclude_extensions": [".jpg", ".jpeg", ".png", ".svg", ".gif"],
        "strip_frontmatter": True,
        "notes": "07-streaming holds many .py files (about 46); review how much code noise they add.",
    },
    "S07": {
        "name": "Luna Reyes et al. - Data Analytics for Public Policy, section 1.4",
        "licence": "CC BY-NC-SA 4.0",
        "path": "S07/S07_LunaReyesEtAl_DataAnalyticsPublicPolicy_Sec1-4.pdf",
        "format": "pdf",
        "extractor": "pdfplumber",
        "source_type": "textbook",
        "page_ranges": None,
        # The PDF is a browser printout: every page starts with a line like
        # "10/6/26, 2:20 PM 1.4 Data Quality and Data Types - ..." that must be removed.
        "strip_line_patterns": [
            r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\b.*$",
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
# Used later to label each chunk with a topic.
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
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 100        # characters shared between neighbouring chunks

# Metadata fields written for every chunk (see proposal Table 3.10)
CHUNK_METADATA_FIELDS = [
    "chunk_id",
    "source_id",
    "source_name",
    "licence",
    "source_type",   # textbook | lecture_slides | wiki | course_notes | code
    "section",       # nearest heading
    "page",          # page number for PDFs, file path for Markdown/code
    "topic",
    "keywords",
    "difficulty",
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
