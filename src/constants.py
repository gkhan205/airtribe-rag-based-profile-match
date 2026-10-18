"""All project constants in one place: paths, model, chunking, extraction, scoring, evaluation, dataset.

The skill taxonomy is large reference data and lives in skills.py.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (anchored at the repository root so commands work from any directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESUMES_DIR = ROOT / "resumes"
JD_DIR = ROOT / "job_descriptions"
INDEX_DIR = ROOT / "index"
OUTPUTS_DIR = ROOT / "outputs"
GROUND_TRUTH = JD_DIR / "ground_truth.json"
LABELS_FILE = "_labels.json"  # inside RESUMES_DIR; evaluation-only metadata

# Index artifact file names (inside an index dir)
FAISS_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "resumes_meta.json"
CONFIG_FILE = "config.json"

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".md")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
ALT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# bge models expect this instruction on queries (not on passages)
QUERY_INSTRUCTIONS = {DEFAULT_MODEL: "Represent this sentence for searching relevant passages: "}
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNKERS = ("section", "fixed")
DEFAULT_CHUNKER = "section"
MAX_CHUNK_CHARS = 1200        # section chunks longer than this are sub-split
CHUNK_OVERLAP = 150
FIXED_CHUNK_CHARS = 800       # 'fixed' baseline strategy
FIXED_CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 30          # drop tiny fragments
MAX_HEADER_LEN = 45           # longer lines are never treated as section headers

# Header line (lower-cased) -> canonical section
SECTION_ALIASES: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "executive summary", "career overview", "profile",
                "professional profile", "objective", "career objective", "overview", "about me", "career focus",
                "executive profile", "summary of qualifications"],
    "skills": ["skills", "technical skills", "core qualifications", "qualifications", "highlights",
               "skill highlights", "core competencies", "competencies", "areas of expertise", "expertise",
               "computer skills", "key skills", "skills summary", "strengths"],
    "experience": ["experience", "work experience", "professional experience", "work history",
                   "employment history", "employment", "relevant experience", "career history",
                   "experience and accomplishments"],
    "education": ["education", "education and training", "academic background", "academic qualifications",
                  "educational background", "training", "education & training"],
    "certifications": ["certifications", "certification", "licenses", "licenses and certifications",
                       "certifications and licenses", "credentials", "professional development"],
    "projects": ["projects", "key projects", "selected projects"],
    "accomplishments": ["accomplishments", "achievements", "awards", "honors", "awards and honors",
                        "presentations", "publications"],
    "other": ["additional information", "interests", "languages", "affiliations", "professional affiliations",
              "volunteer experience", "volunteer work", "activities", "references", "military experience"],
}

# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MAX_CAREER_YEARS = 45         # sanity cap for date ranges and "N years" mentions
SKILL_YEARS_WINDOW = (25, 60)  # chars before / after "N years" searched for the skill it applies to
HEADLINE_MAX_CHARS = 80
EDU_LINE_MAX_CHARS = 120
NOT_A_FIELD = {"degree", "status", "city", "state"}

DEGREES = [  # (level, label, regex) highest first
    (5, "PhD", r"ph\.?\s?d|doctor(?:ate| of)"),
    (4, "Master", r"master'?s?|m\.?b\.?a|m\.s\.|m\.a\.|msc|m\.ed|mfa"),
    (3, "Bachelor", r"bachelor'?s?|b\.?s\.?c?|b\.a\.?|bba|b\.tech|b\.e\."),
    (2, "Associate", r"associate'?s? (?:degree|of)|a\.a\.s?|aas|associate"),
    (1, "Diploma/Certificate", r"high school diploma|diploma|ged|certificate"),
]
JD_MIN_DEGREE_LEVEL = 3       # JD degree requirements below Bachelor are ignored

# ---------------------------------------------------------------------------
# Matching & scoring
# ---------------------------------------------------------------------------
DEFAULT_K = 10
QUERY_K = 5                   # default hits for `main.py query`
MODES = ("hybrid", "semantic", "keyword")
# Score = 100 * sum(weight * component); components are all in [0, 1]
DEFAULT_WEIGHTS = {"semantic": 0.55, "skills": 0.25, "keyword": 0.10, "qualification": 0.10}
# bge-small cosine similarities between a JD and resume chunks fall roughly in
# [0.50, 0.80]; this maps them onto [0, 1] so scores are comparable across queries.
SEMANTIC_FLOOR, SEMANTIC_CEIL = 0.50, 0.80
KEYWORD_MODE_BM25_WEIGHT = 0.5  # keyword-only ablation: 0.5 * BM25 + 0.5 * skill coverage
SEMANTIC_MAX_WEIGHT = 0.7     # resume semantic = 0.7 * best chunk + 0.3 * mean(top-3 chunks)
SEMANTIC_TOP_N = 3
CANDIDATE_CHUNKS = 100        # chunks pulled from FAISS and from BM25 before per-resume aggregation
MUST_SKILL_BOOST = 2          # must-have skills repeated in the BM25 query

# Skill-coverage weights
MUST_SKILL_WEIGHT = 2.0
REQUIRED_SKILL_WEIGHT = 1.0
PREFERRED_SKILL_WEIGHT = 0.5

# Qualification fit = EXP_FIT_WEIGHT * experience fit + (1 - EXP_FIT_WEIGHT) * degree fit
EXP_FIT_WEIGHT = 0.7
DEGREE_FIT_LOWER = 0.5        # has a degree, but below the one asked for
DEGREE_FIT_NONE = 0.25        # no degree found

# Reasoning / output
STRONG_SEMANTIC, MODERATE_SEMANTIC = 0.6, 0.35
N_EXCERPTS = 3
EXCERPT_MAX_CHARS = 300
EXCERPT_MIN_LINE_CHARS = 25
REASONING_MAX_SKILLS = 6
REASONING_MAX_MISSING = 4
JD_SKILL_YEARS_WINDOW = 60    # chars after "N years" in a JD searched for the skill it applies to

BM25_STOPWORDS = set("""a an and the of to in for on with or at by from as is are be we our you your will
this that it its into across using use per who all any must have has nice plus preferred required
requirements responsibilities experience years year strong""".split())

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
METRICS = ["P@5", "P@10", "R@10", "MRR", "nDCG@10"]
EVAL_K = 10
RELEVANT_GRADE = 1            # grade >= this counts as relevant for P/R/MRR
LATENCY_RUNS = 50

# ---------------------------------------------------------------------------
# Dataset download (scripts/fetch_resumes.py)
# ---------------------------------------------------------------------------
HF_DATASET = "opensporks/resumes"  # HuggingFace mirror of Kaggle snehaanbhawal/resume-dataset (CC0)
HF_TREE_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/data/data/{{cat}}"
HF_FILE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{{path}}"
FETCH_SEED = 42
FETCH_PER_CATEGORY = 5
FETCH_CATEGORIES = [
    "INFORMATION-TECHNOLOGY", "ENGINEERING", "HR", "FINANCE", "ACCOUNTANT",
    "SALES", "DESIGNER", "HEALTHCARE", "TEACHER", "BANKING",
]
# Synthetic names (source resumes are anonymized)
FIRST_NAMES = ["James", "Maria", "Ahmed", "Priya", "Chen", "Sofia", "Daniel", "Aisha", "Lucas", "Emma",
               "Omar", "Hannah", "Ravi", "Grace", "Mateo", "Fatima", "Noah", "Yuki", "Ethan", "Zara",
               "Leo", "Amara", "Ivan", "Chloe", "Samuel", "Leila", "Kenji", "Olivia", "Diego", "Nadia",
               "Arjun", "Isabel", "Tariq", "Mia", "Victor", "Sara", "Hugo", "Ana", "Felix", "Rosa",
               "Kwame", "Elena", "Marco", "Ines", "Jonah", "Lina", "Pavel", "Ruth", "Andre", "Maya"]
LAST_NAMES = ["Smith", "Garcia", "Khan", "Patel", "Wang", "Rossi", "Miller", "Bello", "Silva", "Brown",
              "Haddad", "Fischer", "Iyer", "Kim", "Lopez", "Rahman", "Wilson", "Tanaka", "Clark", "Ali",
              "Moreau", "Okafor", "Petrov", "Dubois", "Turner", "Nasser", "Sato", "Evans", "Torres", "Hassan",
              "Mehta", "Costa", "Aziz", "Young", "Novak", "Cohen", "Martin", "Pereira", "Weber", "Diaz",
              "Mensah", "Popescu", "Bianchi", "Ferreira", "Reed", "Saleh", "Volkov", "Adler", "Moore", "Shah"]
