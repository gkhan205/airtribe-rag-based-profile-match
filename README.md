`

# RAG-Based Resume ↔ Job Matching

A local RAG system that indexes resumes into a **FAISS** vector store and ranks candidates against a job description. Ranking combines **hybrid search** (semantic embeddings + BM25 + critical-skill matching), a 0–100 score, must-have filters and rule-based match reasoning. Everything runs offline once the model is downloaded, so no API keys are needed.

```
resumes/*.pdf ──► file_tools.read_file ──► section-aware chunker ──► bge-small embeddings ──► FAISS IndexFlatIP
                                  └────► metadata extractor (name, skills, years, education) ──► resumes_meta.json

job description ──► JD parser (skills, must-haves, min years, degree)
                ├─► query embedding ─► FAISS top-100 chunks ─┐
                └─► BM25 top-100 chunks ──────────────────────┴─► candidate resumes
                                                                   ├─ semantic (max + top-3 mean, calibrated)
                                                                   ├─ skill coverage (must-haves weighted 2x)
                                                                   ├─ BM25 keyword score
                                                                   ├─ qualification fit (experience, degree)
                                                                   └─ must-have filter ─► score 0-100 + reasoning ─► JSON
```

## Project layout

```
main.py                    # single entry point: build | query | show | match | evaluate
requirements.txt
scripts/
  fetch_resumes.py         # one-off: downloads the resume sample into resumes/
src/
  constants.py             # every tunable/constant: paths, model, chunking, section headers, degrees,
                           #   scoring weights & thresholds, evaluation and dataset settings
  file_tools.py            # file system tools (Milestone 1): list_files, file_info, read_file, rel
  skills.py                # skill taxonomy (~100 skills, 8 domains) + alias normalization
  resume_rag.py            # Part A: section chunking, metadata extraction, embeddings, FAISS index (ResumeRAG)
  job_matcher.py           # Part B: JD parsing, hybrid retrieval, scoring, filters, reasoning (JobMatcher)
  evaluation.py            # P@5, P@10, R@10, MRR, nDCG@10 vs ground truth; latency measurement
  analysis.ipynb           # experiments, ablations, accuracy and latency analysis
resumes/                   # 50 resume PDFs (10 categories x 5) + _labels.json (category, evaluation only)
job_descriptions/          # 8 job descriptions + ground_truth.json (graded relevance)
index/                     # generated: faiss.index, chunks.jsonl, resumes_meta.json, config.json
outputs/                   # generated: match JSON per JD
```

## Setup

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
```

Python 3.12 is used because faiss-cpu and torch wheels for 3.14 are not reliable yet. Plain `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt` works too. The embedding model `BAAI/bge-small-en-v1.5` (~130 MB) downloads on first run.

## Usage

```bash
source .venv/bin/activate

python main.py                     # list commands

# 0. (already done) download the dataset
python scripts/fetch_resumes.py --per-category 5

# 1. Part A: build the index
python main.py build
python main.py show resumes/chen_wang.pdf                   # inspect metadata + chunks of one resume
python main.py query "registered nurse EMR patient care" -k 5

# 2. Part B: match job descriptions
python main.py match job_descriptions/staff_accountant.txt                 # full JSON to stdout
python main.py match job_descriptions/*.txt --out outputs/                 # summary + one JSON per JD
python main.py match job_descriptions/it_systems_admin.txt --must "Python:5" --must Linux
python main.py match --text "Graphic designer with Photoshop and branding" --k 5 --no-jd-must
python main.py match job_descriptions/teacher.txt --mode semantic          # ablation: semantic | keyword | hybrid

# 3. Metrics
python main.py evaluate
jupyter notebook src/analysis.ipynb
```

## Output format

```json
{
  "job_description": "Staff Accountant ...",
  "top_matches": [
    {
      "candidate_name": "Omar Haddad",
      "resume_path": "resumes/omar_haddad.pdf",
      "match_score": 83,
      "matched_skills": ["Financial Reporting", "General Ledger", "Reconciliation", "Accounts Payable", "..."],
      "relevant_excerpts": ["[Experience] Ensure that all monthly balance sheet reconciliations are completed ..."],
      "reasoning": "Strong semantic match for 'Staff Accountant' in Experience (headline: Senior Finance Manager); covers 8/9 required skills (...); missing Tax; ~8.1 yrs experience (meets 3+ asked); Master in Business Administration.",
      "missing_skills": ["Tax"],
      "matched_sections": ["experience"],
      "experience_years": 8.1,
      "education": "Master",
      "score_breakdown": {"semantic": 0.794, "skills": 0.792, "keyword": 0.991, "qualification": 1.0}
    }
  ],
  "parsed_requirements": {"must_haves": ["3+ years Accounting"], "...": "..."},
  "filtered_out": [{"candidate_name": "Ravi Iyer", "reasons": ["missing must-have skill Accounting"]}],
  "stats": {"timings_ms": {"embed_query": 20.8, "faiss_search": 0.05, "total": 54.1}}
}
```

The first six fields of each match are the required format. The rest are extras.

## Design choices

All tunables below (chunk sizes, weights, calibration range, thresholds) live in [`src/constants.py`](src/constants.py). The skill taxonomy is reference data and lives in `src/skills.py`.

**Chunking (Part A).**

- Header lines are mapped onto canonical sections: summary, skills, experience, education, certifications, projects, accomplishments and other. Text before the first header is labelled `profile`.
- Each section is one chunk. Sections over 1200 chars are split on line boundaries with 150 chars of overlap.
- Resumes with no recognizable headers fall back to a sliding window.
- Each chunk is embedded with its section name prepended (`"Experience: ..."`).

**Metadata.**

- *Skills*: taxonomy matches, with the Skills/Summary sections searched first.
- *Experience years*: merged employment date ranges (`Jan 2015 to Current`, `05/2013 - 08/2014`, `2010 - 2012`). "Current" resolves to the latest year in the resume. Stated "X years experience" is used when it is larger.
- *Per-skill years*: from "N years … <skill></skill>" patterns.
- *Education*: highest degree and field of study.

FAISS stores only vectors, so the metadata lives in JSON sidecars joined by vector position. The metadata drives filtering and scoring.

**Vector store.** `faiss.IndexFlatIP` on L2-normalized vectors gives exact cosine similarity. At this scale exact search takes ~0.05 ms. The notebook shows HNSW as the upgrade path for 100k+ chunks.

**Hybrid retrieval (Part B).**

- The candidate pool is the union of the FAISS top-100 chunks and the BM25 top-100 chunks.
- Each candidate resume gets four component scores:
  - **semantic**: `0.7·max + 0.3·mean(top-3)` chunk cosine, calibrated from [0.50, 0.80] to [0, 1]
  - **skill coverage**: required skills, with must-haves weighted ×2 and preferred skills ×0.5
  - **BM25**: normalized
  - **qualification fit**: experience vs. the asked minimum, plus degree level
- Final score: `100 × (0.55·sem + 0.25·skills + 0.10·bm25 + 0.10·qual)`.

**Must-have filters.**

- Auto-detected from JD lines containing *must / required*, e.g. "Must have: 3+ years of Accounting experience" becomes `3+ years Accounting`.
- Can also be passed with `--must "Skill[:years]"` or `--must years:N`.
- A skill-years check uses explicit per-skill years when the resume states them. Otherwise it uses total experience as a proxy.
- Rejected candidates are listed with reasons.

**Reasoning.** Rule-based (deterministic, no LLM). It is built from the matched sections, the matched and missing skills, experience vs. requirement, and education.

## Results (from `src/analysis.ipynb`)

| Mode                         | P@5             | P@10*           | R@10            | MRR | nDCG@10         |
| ---------------------------- | --------------- | --------------- | --------------- | --- | --------------- |
| Semantic only                | 0.850           | 0.588           | 0.904           | 1.0 | 0.881           |
| Keyword only (BM25 + skills) | 0.800           | 0.550           | 0.823           | 1.0 | 0.858           |
| **Hybrid**             | **0.875** | **0.600** | **0.915** | 1.0 | **0.906** |

\*P@10 is capped by the 5–11 relevant resumes per JD.

- **Latency:** p50 ≈ 54 ms end-to-end per query on CPU (query embedding ≈ 21 ms, FAISS ≈ 0.05 ms). Index build ≈ 24 s for 50 resumes.
- **Embedding model:** bge-small beats MiniLM-L6, especially in semantic-only mode (nDCG 0.881 vs 0.821).
- **Chunking:** fixed-size chunking ranks about as well on these short resumes. Section-aware chunking's benefit is explainability.

See the notebook's findings section for the full write-up.

## Dataset & ethics note

- **Resumes:** 50 PDFs sampled (seed 42, 5 each from IT, Engineering, HR, Finance, Accountant, Sales, Designer, Healthcare, Teacher and Banking) from the Kaggle [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (CC0). They come via its HuggingFace mirror [`opensporks/resumes`](https://huggingface.co/datasets/opensporks/resumes).
- **Names:** the source resumes are anonymized, so **candidate names are synthetic** and assigned deterministically. They do not refer to real people.
- **Job descriptions:** the 8 JDs were written for this project.
- **Ground truth:** graded by category (2), plus hand-picked adjacent roles (1).
