# 🤖 Multi-Agent Analytics Assistant (Human-in-the-Loop)

Ask questions about an e-commerce database in plain English and get real answers —
with a human check on everything the AI generates before it runs.

Under the hood, a **supervisor** reads each question and hands it to the right
specialist: a **SQL agent** that writes its own schema-aware queries, a **pandas
agent** that runs statistical analysis with automated data-quality checks, a
**charts agent** that fetches data and plots it, or a **direct** answer when the
question can be answered from the conversation so far. Before any generated SQL,
analysis code, or chart code executes, the system pauses and asks you to approve,
revise, or skip it.

Built with LangGraph, with an interactive Streamlit chat UI.

> Ask a question → the supervisor picks the right agent → that agent explores the
> data and proposes SQL or Python → **you review, revise, or approve it** → it runs
> safely and answers in plain language.

---

## ✨ What it does

- **Routes intelligently.** A supervisor classifies each question and sends it to
  one of four handlers — `sql`, `pandas`, `viz`, or `direct` — and the UI shows you
  which agent it picked and why.
- **Writes its own SQL.** The SQL agent inspects the database schema first, then
  writes multi-table queries grounded in the real tables and columns instead of
  guessing names.
- **Does real statistics.** For questions SQL can't cleanly answer (correlations,
  distributions), the pandas agent fetches the data, runs **automated data-quality
  checks** — nulls, duplicates, type mismatches, outliers — and writes analysis
  code that accounts for what it found.
- **Draws charts.** The charts agent fetches the right columns, generates matplotlib
  code, and returns both the image and the numbers behind it.
- **Keeps a human in the loop.** Every agent that generates code pauses and shows
  you the SQL, the analysis code, or the chart code (plus the data-quality report)
  so you can **approve**, **revise** with feedback, or **skip** — before anything runs.
- **Remembers the conversation.** Follow-up questions like "what was that Pearson
  value again?" are answered instantly from earlier in the chat instead of
  re-running the analysis.
- **Runs safely.** SQL execution is read-only (SELECT only), and model-generated
  pandas code runs in a scoped, sandboxed namespace.

---

## 🏗️ How it works

Each specialist is a self-contained LangGraph sub-graph nested inside the
supervisor. When a specialist pauses for approval, that interrupt surfaces up
through the supervisor to you, and your decision resumes it back down.

```
                          ┌──────────────┐
        user question ───►│  SUPERVISOR  │  reads the question, picks a route
                          └──────┬───────┘
              ┌──────────────────┼──────────────────┬──────────────┐
              ▼                  ▼                  ▼              ▼
       ┌────────────┐    ┌──────────────┐   ┌────────────┐  ┌──────────┐
       │ SQL AGENT  │    │ PANDAS AGENT │   │ CHARTS     │  │  DIRECT  │
       │            │    │ + data-QA    │   │ AGENT      │  │  answer  │
       │ explore →  │    │ fetch → QA → │   │ fetch →    │  │ (recall) │
       │ propose SQL│    │ propose code │   │ plot code  │  └──────────┘
       └─────┬──────┘    └──────┬───────┘   └─────┬──────┘
             └──────────────────┼─────────────────┘
                                ▼
                  ┌───────────────────────────────┐
                  │      HUMAN APPROVAL  (⏸)       │  approve / revise / skip
                  │  review the SQL / code + QA    │  (nested interrupt → supervisor)
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │            EXECUTE             │  read-only SQL / sandboxed pandas
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │            ANSWER              │  clear, plain-language response
                  └───────────────────────────────┘
```

A couple of design choices that make this hold together:

- **Each question fetches its own data.** Agents reset their fetch state at the
  start of every question, so a follow-up never accidentally analyzes the previous
  question's data.
- **The supervisor and direct agent see the full conversation** (for routing and
  recall), while each worker agent is pinned to the *current* question — so the SQL
  agent doesn't drag an earlier question's query into a new one.

---

## 🛠️ Tech stack

- **LangGraph** — multi-agent orchestration, routing, nested interrupts, checkpointing
- **LangChain + OpenAI** — the models that route, write SQL/code, and interpret results
- **SQLAlchemy** — database access
- **SQLite** (Olist e-commerce dataset) — the data store
- **pandas + NumPy** — data fetching, quality checks, and analysis
- **matplotlib** — chart generation
- **Streamlit** — the interactive chat UI

---

## 📦 Setup

### 1. Clone and install

```bash
git clone https://github.com/muradohi/sql-analytics-agent.git
cd sql-analytics-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Kaggle, unzip the CSVs into a `data/` folder, then load them into the database:

```bash
python load_data.py
```

This builds the tables (`orders`, `order_items`, `products`, `customers`,
`order_reviews`) into a local SQLite database (`olist.db`).

### 3. Add your API key

Create a `.env` file in the project root (it's gitignored):

```
OPENAI_API_KEY=sk-your-key-here
```

Non-secret settings (the model name, DB connection) live in `config/config.yaml`:

```yaml
llm:
  model: gpt-5-mini

database:
  drivername: sqlite
  database: olist
```

### 4. Run it

The Streamlit app (recommended):

```bash
streamlit run app.py
```

Or drive the supervisor from the command line:

```bash
python -m agents.supervisor
```

> Run from the project root so the `agents` package imports resolve.

---

## 💬 Examples

**A database question → SQL agent**
> "How many orders were cancelled?"

The supervisor routes to the SQL agent. It inspects the schema, proposes a
`COUNT` query for your approval, runs it once approved, and answers: *625 orders
were cancelled.*

**A statistical question → pandas agent**
> "Is there a correlation between price and review score?"

Routed to the pandas agent. It fetches price and review scores, runs
data-quality checks (flagging duplicates and outliers), proposes pandas code that
handles them, and after approval reports: *no meaningful correlation — Pearson ≈
0.05, dropping to ≈ 0.02 once outliers are removed.*

**A chart request → charts agent**
> "Plot the top 5 product categories by number of orders"

Routed to the charts agent. It fetches the ranked categories, proposes matplotlib
code for a bar chart, and after approval shows the plot along with the underlying
numbers.

**A follow-up → direct answer (from memory)**
> "What was the Pearson value again?"

No new query needed — the supervisor recognizes the answer is already in the
conversation and recalls it directly.

---

## 📂 Project structure

```
sql-analytics-agent/
├── app.py                       # Streamlit chat UI (entry point)
├── agents/
│   ├── __init__.py              # re-exports the supervisor graph + engine
│   ├── supervisor.py            # supervisor: routing + nested agents
│   ├── agent_sql_app.py         # SQL agent (schema-aware, HITL approval)
│   ├── agent_pandas_app.py      # pandas agent (data-quality checks + HITL)
│   └── agent_viz_app.py         # charts agent (fetch → plot → HITL)
├── config/
│   └── config.yaml              # model + database settings
├── load_data.py                 # loads the Olist CSVs into SQLite
├── data/                        # Olist CSVs (gitignored)
├── requirements.txt
├── .env                         # secrets (gitignored)
└── README.md
```

---

## 🧠 Design decisions

- **Determinism where it belongs.** Data loading and data-quality checks are plain
  code, not LLM calls — they have exact right answers and must be reliable. The
  model is used only where judgment is needed: routing, writing queries and code,
  and interpreting results.
- **Schema discovery over hardcoding.** Agents inspect the database themselves, so
  the system adapts to the schema rather than depending on hardcoded table names.
- **A real approval gate, not a demo.** No agent runs generated SQL or Python
  without a human seeing it first — and the revise loop feeds your feedback back to
  the model so it can correct course.
- **Memory vs. task, deliberately separated.** The supervisor holds the
  conversation for routing and recall; the worker agents are scoped to the current
  question so old context can't leak into a new query.
- **Layered safety.** Read-only SQL, sandboxed pandas execution, and the human gate
  together bound what generated code can do.

---

## 🔮 Roadmap

- [ ] "Bring your own OpenAI key" so anyone can use the deployed app freely
- [ ] Multi-task routing (splitting a compound question across agents)
- [ ] A retrieval (RAG) agent over review text, fused with structured results
- [ ] Persistent conversation memory across sessions

---