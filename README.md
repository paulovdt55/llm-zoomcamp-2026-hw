# LLM Zoomcamp 2026 — Homework & Code

My solutions and code for the [LLM Zoomcamp 2026](https://courses.datatalks.club/llm-zoomcamp-2026/)
by [DataTalks.Club](https://datatalks.club/).

The LLM Zoomcamp is a free, hands-on course about building real-life applications
with Large Language Models — covering Retrieval-Augmented Generation (RAG),
vector and hybrid search, embeddings, AI agents, function calling, evaluation,
and monitoring.

This repository will hold **all modules of the course**. Each module lives in its
own folder and is added as the course progresses.

## Repository structure

This is a monorepo: one repository, one folder per module.

```
llm-zoomcamp/
├── README.md            # this file
├── .gitignore
└── 01-agentic-rag/      # Module 1 — Agentic RAG
    ├── 01-fetch-files.ipynb
    ├── 02-index-search.ipynb
    ├── 03-rag-helper.ipynb
    ├── 04-chunking.ipynb
    ├── 05-rag-chunking.ipynb
    ├── 06-agent.ipynb
    ├── documents.json
    ├── chunks.json
    ├── pyproject.toml
    └── uv.lock
```

## Modules

| Module | Topic | Status |
| ------ | ----- | ------ |
| 01 | Agentic RAG | Done |
| 02 | _to be added_ | Planned |
| 03 | _to be added_ | Planned |
| 04 | _to be added_ | Planned |
| 05 | _to be added_ | Planned |
| 06 | _to be added_ | Planned |

The module list will be filled in as the course releases each week's material.

## Tooling

- **Python** managed with [uv](https://github.com/astral-sh/uv) — each module has
  its own `pyproject.toml`, `uv.lock`, and isolated `.venv`.
- **Notebooks** run in VS Code / Jupyter.
- Libraries used so far: `minsearch`, `gitsource`, `openai`, `toyaikit`.

### Running a module

```bash
cd 01-agentic-rag
uv sync          # recreate the environment from the lock file
```

Each module's notebooks are numbered and meant to run in order (e.g. `01` produces
data that later notebooks load).

## Notes

- `.env` files and virtual environments (`.venv/`) are intentionally excluded via
  `.gitignore` and never committed.
- Generated artifacts such as `documents.json` and `chunks.json` are committed
  here for reproducibility, so the notebooks can be reviewed without re-running
  every upstream step.

## Links

- Course: https://courses.datatalks.club/llm-zoomcamp-2026/
- Official course repository: https://github.com/DataTalksClub/llm-zoomcamp
- DataTalks.Club: https://datatalks.club/
