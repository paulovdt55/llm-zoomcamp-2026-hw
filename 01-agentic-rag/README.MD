# LLM Zoomcamp 2026 — Module 1: Agentic RAG (Homework)

Build a RAG system from scratch over the course lessons, then make it agentic. The knowledge base is the LLM Zoomcamp lesson pages themselves, pulled directly from the course GitHub repository.

## Setup

```bash
uv add gitsource
export OPENAI_API_KEY=...   # recommended model: gpt-5.4-mini (any provider works)
```

## Data preparation

Pull the lesson pages from the repo, pinned to commit `8c1834d` so everyone works with identical data.

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
# each doc -> {"filename": ..., "content": ...}
```

---

## Q1. How many lesson pages

Count how many lesson pages ended up in `documents` after the preparation step.

```python
len(documents)
```

Options: 24 · 72 · 240 · 720

---

## Q2. Indexing and searching

Index the documents with `minsearch`: make `content` a **text** field and `filename` a **keyword** field. Then run the query and read the `filename` of the first result.

```python
from minsearch import Index

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

query = "How does the agentic loop keep calling the model until it stops?"
results = index.search(query, num_results=5)
results[0]["filename"]
```

Options: `01-agentic-rag/lessons/03-rag.md` · `01-agentic-rag/lessons/14-agentic-loop.md` · `04-evaluation/lessons/13-llm-as-judge.md` · `06-best-practices/lessons/02-hybrid-search.md`

---

## Q3. RAG

Build a RAG assistant on top of the Q2 index and answer the same query. Reuse the course `rag_helper.py`:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
```

`RAGBase` was written for the FAQ schema (`section`/`question`/`answer`); our docs use `filename`/`content`. Either implement the RAG flow yourself, or subclass `RAGBase` and override `search` (to use this index) and `build_context`.

Modify the helper so the `llm` method returns the **whole response** (not just text), and `rag` returns both the answer and usage. Read input tokens from `response.usage.input_tokens` / `prompt_tokens`.

Question: how many **input (prompt) tokens** were sent for this request?

Options: 700 · 7000 · 70000 · 700000

---

## Q4. Chunking

Lesson pages are long, which hurts retrieval precision. Split each page into smaller overlapping windows with `chunk_documents`:

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
len(chunks)
```

A window of `size` characters slides across the text in steps of `step`. Since `step < size`, consecutive chunks overlap by `size - step` (1000) characters. Each chunk keeps `filename` and adds `start` (offset) and `content`.

Question: how many chunks do you get?

Options: 70 · 295 · 1100 · 4500

---

## Q5. RAG with chunking

Index the Q4 chunks the same way (`content` = text, `filename` = keyword), point the RAG at the chunk index, and answer the same query — reading input tokens as in Q3.

```python
chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)
# rerun the RAG flow against chunk_index, compare input tokens vs. Q3
```

Question: how many **fewer** input tokens does the chunked version send vs. Q3?

Options: about the same · 3× fewer · 10× fewer · 30× fewer

---

## Dependencies

- `gitsource` — download files from a GitHub repo + chunking helper
- `minsearch` — in-memory text index
- An LLM client (OpenAI `gpt-5.4-mini` recommended)
- `toyaikit` — for the agent step (Q6, beyond this scope)

## References

- [Homework](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/01-agentic-rag/homework.md)
- [Course repository](https://github.com/DataTalksClub/llm-zoomcamp)
- [Submit answers](https://courses.datatalks.club/llm-zoomcamp-2026/hw1)
