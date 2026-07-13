# LLM Zoomcamp 2026 — Homework 4: Evaluation

Solução do [Homework 4: Evaluation](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/04-evaluation/homework.md) do módulo **04-evaluation** do [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp).

Notebook: [`hw4_evaluation.ipynb`](hw4_evaluation.ipynb)

## Contexto

No Homework 2 foi construída busca por palavra-chave (keyword), vetorial e híbrida sobre as lições do curso, sem responder qual delas é melhor. Este homework gera um dataset de ground truth e usa esse dataset para medir e comparar as três abordagens de busca com métricas (Hit Rate e MRR), em vez de depender de intuição.

A base de conhecimento é o próprio conteúdo do curso: cada módulo tem uma pasta `lessons/` com páginas markdown numeradas, obtidas diretamente do GitHub no commit fixo `8c1834d` (72 páginas, para todo mundo trabalhar com os mesmos dados).

## O que o notebook faz

1. **Setup** — instala dependências (`openai`, `pydantic`, `python-dotenv`, `pandas`, `gitsource`, `minsearch`, `sentence-transformers`) e baixa `rag_helper.py` / `evaluation_utils.py` do repositório do curso.
2. **Carregar as páginas de lição** — via `gitsource.GithubRepositoryDataReader`, fixando o commit `8c1834d`.
3. **Geração de perguntas (ground truth)** — usa um LLM (`gpt-5.4-mini` via OpenAI, structured output) para gerar 5 perguntas por página de lição, rodando primeiro só nas 3 primeiras páginas para medir custo/tokens.
4. **Ground truth completo** — carrega o arquivo `ground-truth.csv` (360 perguntas, já gerado pelo material do curso) em vez de reprocessar as 72 páginas.
5. **Chunking** — reaproveita `chunk_documents` (do HW2) para dividir as páginas em 295 chunks sobrepostos (`size=2000`, `step=1000`).
6. **Índices de busca** — reconstrói `text_search` (minsearch `Index`) e `vector_search` (minsearch `VectorSearch` + embeddings `all-MiniLM-L6-v2` via sentence-transformers) sobre os chunks, indexados por `filename`.
7. **Métricas de avaliação** — `hit_rate`, `mrr` e `evaluate`, reaproveitados das aulas do módulo, adaptados para comparar pelo campo `filename`.
8. **Hybrid search com RRF** — reaproveita o `rrf` do HW2 e testa o parâmetro `k` do Reciprocal Rank Fusion em `[1, 50, 100, 200]` para ver qual valor maximiza o MRR.

## Perguntas respondidas

| # | Pergunta | Como é calculada no notebook |
|---|----------|-------------------------------|
| Q1 | Média de input tokens ao gerar perguntas para as 3 primeiras páginas | Sessão 2 |
| Q2 | Primeiro resultado do `text_search` para a 1ª pergunta do ground truth | Sessão 6 |
| Q3 | Primeiro resultado do `vector_search` para a mesma pergunta | Sessão 6 |
| Q4 | Hit Rate do `text_search` sobre todo o ground truth | Sessão 7 |
| Q5 | MRR do `vector_search` sobre todo o ground truth | Sessão 7 |
| Q6 | Melhor valor de `k` no `hybrid_search` (RRF), entre 1, 50, 100 e 200 | Sessão 8 |

## Como rodar

1. Crie um arquivo `.env` na raiz do projeto com sua chave da OpenAI:

   ```
   OPENAI_API_KEY=sk-...
   ```

2. Abra `hw4_evaluation.ipynb` em Jupyter e execute as sessões em ordem (0 a 9).

3. Ao final, a Sessão 9 imprime um resumo com as respostas calculadas para as 6 perguntas do homework, prontas para submeter em [courses.datatalks.club/llm-zoomcamp-2026/homework/hw4](https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw4).

> Custo estimado: poucos centavos em créditos de API (só a geração de perguntas via LLM usa a API paga; a busca vetorial roda localmente com `sentence-transformers`).

## Referências

- [Instruções oficiais do homework](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/cohorts/2026/04-evaluation/homework.md)
- [Lições do módulo 04-evaluation](https://github.com/DataTalksClub/llm-zoomcamp/tree/main/04-evaluation/lessons)
- [Repositório do curso](https://github.com/DataTalksClub/llm-zoomcamp)
