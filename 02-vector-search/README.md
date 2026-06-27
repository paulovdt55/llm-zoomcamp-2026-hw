# LLM Zoomcamp 2026 — Module 2: Vector Search (Homework)

Turn text into vectors and search by similarity, then combine vector search with keyword search. The knowledge base is the LLM Zoomcamp lesson pages themselves, pulled directly from the course GitHub repository. We skip the RAG part and focus solely on search.

Unlike the module, this homework uses a lightweight **ONNX `Embedder`** instead of `sentence-transformers`. Both produce identical vectors, but the ONNX runtime needs no PyTorch and no CUDA — about 30× smaller install, runs anywhere, including a basic Codespace.

## Setup

```bash
mkdir llm-zoomcamp-hw2 && cd llm-zoomcamp-hw2
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter

# helper scripts from the course repo's embed/ directory
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed
wget $PREFIX/download.py    # fetches an ONNX model from HuggingFace
wget $PREFIX/embedder.py    # the Embedder class with an encode interface

uv run python download.py   # defaults to Xenova/all-MiniLM-L6-v2
```

The embedder returns **normalized** 384-dim vectors, so the dot product between two of them is their cosine similarity.

## Data preparation

Pull the lesson pages from the repo, pinned to commit `8c1834d` so everyone works with identical data (72 pages).

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

## Q1. Embedding a query

Embed the query *"How does approximate nearest neighbor search work?"*. The embedder returns 384 numbers — report the first value.

```python
from embedder import Embedder

embed = Embedder()
v = embed.encode("How does approximate nearest neighbor search work?")
v[0]
```

Options: -0.31 · -0.02 · 0.12 · 0.44

---

## Q2. Cosine similarity

Embed the `content` of `02-vector-search/lessons/07-sqlitesearch-vector.md` and compute the cosine similarity with the Q1 query vector. Because vectors are normalized, the dot product **is** the cosine similarity.

```python
page = next(d for d in documents
            if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
page_vec = embed.encode(page["content"])
float(page_vec.dot(v))
```

Options: 0.07 · 0.37 · 0.68 · 0.92

---

## Q3. Chunking and search by hand

A full page covers several topics, which waters down its embedding. Chunk the pages, embed every chunk's `content`, stack the vectors into a matrix `X`, and score the Q1 query against all chunks.

```python
import numpy as np
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
X = np.array([embed.encode(c["content"]) for c in chunks])   # (n_chunks, 384)

scores = X.dot(v)
chunks[int(np.argmax(scores))]["filename"]
```

Which file does the highest-scoring chunk belong to?

Options: `03-embeddings-dataset.md` · `06-rag-vector.md` · `07-sqlitesearch-vector.md` · `09-onnx-embedder.md`

---

## Q4. Vector search with minsearch

Doing it by hand is good for learning, but in practice we use libraries. Use `VectorSearch` from `minsearch`.

```python
from minsearch import VectorSearch

vindex = VectorSearch()
vindex.fit(X, chunks)   # X = chunk embeddings, chunks = metadata

query = "What metric do we use to evaluate a search engine?"
results = vindex.search(embed.encode(query), num_results=5)
results[0]["filename"]
```

> `VectorSearch.search` takes a **vector** (`embed.encode(query)`), not a string.

Options: `02-vector-search/lessons/04-vector-search.md` · `04-evaluation/lessons/05-search-metrics.md` · `04-evaluation/lessons/13-llm-as-judge.md` · `05-monitoring/lessons/04-metrics.md`

---

## Q5. Text search vs vector search

Vector search matches by meaning, keyword search by exact words. Index the same chunks with `Index` from `minsearch` (use `content` as a **text** field) and run both searches.

```python
from minsearch import Index

text_index = Index(text_fields=["content"], keyword_fields=[])
text_index.fit(chunks)

query = "How do I store vectors in PostgreSQL?"
text_results   = text_index.search(query, num_results=5)              # string
vector_results = vindex.search(embed.encode(query), num_results=5)    # vector
```

> Text search (`Index`) takes the **raw string**; vector search (`VectorSearch`) takes the **embedding**. Don't mix them up.

Which file shows up in the **vector** top-5 but not in the **text** top-5?

Options: `01-intro.md` · `02-embeddings.md` · `08-pgvector.md` · `03-orchestration/lessons/05-rag.md`

---

## Q6. Hybrid search

Vector and text search each have strengths and weaknesses, so we merge their two ranked lists into one with **Reciprocal Rank Fusion (RRF)**. RRF ignores the raw scores (which live on different, incomparable scales) and looks only at each document's **position**:

```
RRF(d) = sum over lists of  1 / (k + rank(d))      # k = 60, rank starts at 0
```

A document found by both searches collects a contribution from each list, so it can outrank one that's only strong in a single list.

```python
def rrf(result_lists, k=60, num_results=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["content"])   # unique per chunk
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query = "How do I give the model access to tools?"
text_results   = text_index.search(query, num_results=5)
vector_results = vindex.search(embed.encode(query), num_results=5)

fused = rrf([text_results, vector_results], k=60, num_results=5)
fused[0]["filename"]
```

> The winning file is **not** first in either search on its own — it wins because it ranks high in both. Note: chunks here expose only `filename` and `content`, so the dedup key uses `content` (the homework's `start` field isn't present in this build).

Which file is ranked first after RRF?

Options: `01-agentic-rag/lessons/01-intro.md` · `13-function-calling.md` · `14-agentic-loop.md` · `16-other-frameworks.md`

---

## Dependencies

- `onnxruntime` + `tokenizers` — lightweight ONNX embedding (no PyTorch/CUDA)
- `gitsource` — download files from a GitHub repo + chunking helper
- `minsearch` — in-memory `Index` (text) and `VectorSearch` (vector)
- `numpy` — matrix scoring by hand
- Helper scripts: `download.py` (model fetch), `embedder.py` (`Embedder` class)

## Credits & References

Course and content created by **[Alexey Grigorev](https://www.linkedin.com/in/agrigorev/)** and **[DataTalksClub](https://datatalks.club/)**. All lesson pages and homework are part of the free [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

- [Homework — Module 2: Vector Search](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/02-vector-search/homework.md)
- [ONNX Embedder lesson](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/lessons/09-onnx-embedder.md)
- [Course repository](https://github.com/DataTalksClub/llm-zoomcamp)
- [Submit answers](https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw2)
