# 🤖 Multi-Agent Analytics Assistant with Human-in-the-Loop

A supervisor routed multi-agent system that answers business questions about an
e-commerce database. A **supervisor** classifies each question and routes it to a
specialist: a **SQL agent** that writes its own schema-aware queries, or a
**pandas agent** that runs statistical analysis with automated data-quality
checks. Both specialists pause for **human approval** of the generated SQL or
the generated analysis code before anything executes.

Built with LangGraph.

> Ask a question in plain English → a supervisor decides which agent should handle
> it → that agent explores the data and proposes SQL or Python code → **you review,
> revise, or approve it** → it runs safely and returns a clear answer.

---

## ✨ Features

- **Supervisor routing** — a router classifies each question and delegates it to
  the right specialist agent (SQL, pandas, or a direct answer).
- **SQL agent** — discovers the database schema itself and writes multi-table
  queries grounded in the real schema, rather than hallucinating table/column names.
- **Pandas agent** — for statistical questions (correlations, distributions) that
  SQL can't cleanly answer. It fetches the needed data, runs **automated
  data-quality checks** (nulls, duplicates, type mismatches, outliers), and writes
  pandas code informed by those checks.
- **Human-in-the-loop approval** — both agents pause and show the human the
  generated SQL or code (plus the data-quality report), who can **approve**,
  **revise** (with feedback), or **skip** before execution.
- **Nested interrupts** — each specialist is a sub-graph whose approval pause
  surfaces up through the supervisor for the human, then resumes back down.
- **Safety** — read-only SQL execution, and sandboxed execution of model-generated
  pandas code in a scoped namespace.
- **Interactive Streamlit UI** — shows routing, generated code, results, and the
  approval workflow.

---

## 🏗️ Architecture

```
                          ┌──────────────┐
        user question ───►│  SUPERVISOR  │  classifies + routes the question
                          └──────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                 ▼
        ┌──────────────┐  ┌──────────────┐   ┌───────────┐
        │  SQL AGENT   │  │ PANDAS AGENT │   │  direct   │
        │              │  │ + data-QA    │   │  answer   │
        │ explore →    │  │ fetch → QA → │   └───────────┘
        │ propose SQL  │  │ propose code │
        └──────────────┘  └──────────────┘
                │                │
                ▼                ▼
        ┌───────────────────────────────┐
        │      HUMAN APPROVAL (⏸)        │  approve / revise / skip
        │  reviews SQL or code + QA      │  (nested interrupt → supervisor)
        └───────────────────────────────┘
                │
                ▼
        ┌───────────────┐
        │   EXECUTE      │  read-only SQL / sandboxed pandas
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │    ANSWER      │  clear, plain-language response
        └───────────────┘
```

Each specialist is a self contained LangGraph agent nested inside the supervisor.
Their human-approval interrupts surface through the supervisor so a person reviews
the generated SQL/code before it runs.

---

## 🛠️ Tech Stack

- **LangGraph** — multi-agent orchestration, routing, nested interrupts, checkpointing
- **LangChain + OpenAI** — the LLMs that route, write SQL/code, and answer
- **SQLAlchemy + PyMySQL** — database access
- **MySQL / SQLite** — the data store (Olist e-commerce dataset)
- **pandas + NumPy** — data fetching, quality checks, and analysis
- **Streamlit** — the interactive UI

---

## 📦 Setup

### 1. Clone and install
```bash
git clone https://github.com/muradohi/sql-analytics-agent.git
cd sql-analytics-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up the database
Download the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from Kaggle, unzip the CSVs into a `data/` folder, then load them:
```bash
python load_data.py
```
This creates the tables (`orders`, `order_items`, `products`, `customers`,
`order_reviews`).

### 3. Configure secrets
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
DB_PASSWORD=your-mysql-password
```

### 4. Run
Command-line (the supervisor + both agents):
```bash
python supervisor.py
```
Interactive UI:
```bash
streamlit run app.py
```

---

## 💬 Examples

**A database question → SQL agent:**
> "How many orders were cancelled?"

The supervisor routes to the SQL agent, which inspects the schema, proposes a
query for your approval, and (once approved) runs it and answers.

**A statistical question → pandas agent:**
> "Is there a correlation between product price and review score?"

The supervisor routes to the pandas agent. It fetches price and review scores,
runs data-quality checks (flagging duplicates and outliers), proposes pandas code
that accounts for those issues, and — after your approval — computes the
correlation (with and without outliers) and answers:
> "No meaningful correlation Pearson ≈ 0.05, dropping to ≈ 0.02 after removing
> outliers."

---

## 📂 Project Structure

```
sql-analytics-agent/
├── supervisor.py          # the supervisor: routing + nested agents
├── agent_sql_app.py       # the SQL agent (schema-aware, HITL)
├── agent_pandas_app.py    # the pandas agent (data-QA + HITL)
├── app.py                 # Streamlit UI
├── load_data.py           # loads the Olist dataset
├── requirements.txt
├── .env                   # secrets (gitignored)
└── README.md
```

---

## 🧠 Design Decisions

- **Determinism where it belongs.** Data loading and data-quality checks are plain
  code, not LLM calls — they have exact correct answers and must be reliable. The
  LLM is used only where judgment is needed (routing, writing queries/code,
  interpreting results).
- **Schema discovery over hardcoding.** The SQL agent inspects the database itself,
  so it adapts to the schema rather than relying on hardcoded table names.
- **Human oversight on generated code.** Neither agent executes generated SQL or
  Python without a human reviewing it first — a real approval gate, not just a demo.
- **Safety layers.** Read-only SQL, sandboxed pandas execution, and the human gate
  together limit what generated code can do.

---

## 🔮 Roadmap

- [ ] Multi-task routing (splitting a compound question across both agents)
- [ ] A retrieval (RAG) agent over review text, fused with structured results
- [ ] Charts/visualizations of analysis output
- [ ] Conversation memory across questions

---

## 📄 License

MIT