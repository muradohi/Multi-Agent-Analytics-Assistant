# 🗃️ SQL Analytics Agent with Human in the Loop

An LLM-powered agent that answers business questions about an e-commerce database
by writing its own SQL — with a **human approval gate** on every generated query
before it runs. Built with LangGraph, it explores the schema, composes a query,
pauses for a human to review the actual SQL, and only executes after approval.

> Ask a question in plain English → the agent inspects the database schema →
> writes a SQL query → **you review, revise, or approve it** → it runs and returns
> a clear answer. Includes an interactive Streamlit UI.

---

## ✨ Features

- **Schema-aware SQL generation** — the agent discovers tables and columns itself
  before writing queries, so it grounds SQL in the real schema instead of guessing.
- **Human-in-the-loop approval** — every generated query is shown to the user, who
  can **approve**, **revise** (with natural-language feedback), or **skip** it
  before it touches the database.
- **Read-only safety guard** — only `SELECT` queries are allowed; the agent cannot
  modify data.
- **Self-correcting** — SQL errors are fed back so the agent can fix its query.
- **Interactive Streamlit UI** — review the generated SQL, preview results, and
  approve/revise visually.
- **Runs on a real MySQL database** (the Olist Brazilian e-commerce dataset).

---

## 🏗️ Architecture

The agent is a LangGraph state machine with an explore → propose → approve →
execute → answer pipeline:

```
        START
          │
          ▼
   ┌─────────────┐   explores schema (list_tables, describe_table)
   │  llm_node   │   and proposes a final query
   └─────────────┘
          │
          ▼
   ┌─────────────┐   PAUSES — human reviews the SQL
   │  approval   │ ⏸ approve / revise / skip
   └─────────────┘
       │    │    │
   approve revise skip
       │    │    │
       ▼    │    ▼
   ┌────────┐│   END
   │execute ││
   └────────┘└──► back to llm (rewrite the query)
       │
       ▼
   ┌────────┐
   │ answer │   writes a plain-language answer from the results
   └────────┘
       │
       ▼
      END
```

- **Interrupts** (LangGraph's human-in-the-loop mechanism) create the pause.
- **A checkpointer** persists graph state so the pause can be resumed.
- **A read-only guard** on the execute step restricts queries to `SELECT`.

---

## 🛠️ Tech Stack

- **LangGraph** — agent orchestration, interrupts, checkpointing
- **LangChain + OpenAI** — the LLM that writes SQL and answers
- **SQLAlchemy + PyMySQL** — database access
- **MySQL** — the data store (Olist e-commerce dataset)
- **Streamlit** — the interactive UI
- **Pandas** — data loading and result display

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
`order_reviews`) in a MySQL database named `olist`.

### 3. Configure secrets
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
DB_PASSWORD=your-mysql-password
```

### 4. Run
Command-line version:
```bash
python agent_sql.py
```
Interactive UI:
```bash
streamlit run streamlit_app.py
```

---

## 💬 Example

**Question:** "Which product category has the highest average review score?"

1. The agent inspects `order_reviews`, `order_items`, and `products`.
2. It proposes a multi-table join query — **shown for your approval**.
3. You approve (or revise, e.g. "only include categories with 100+ reviews").
4. It runs the query and answers:
   > The highest-rated category is **cds_dvds_musicais** at about 4.6 stars,
   > though with a small number of reviews. Among high-volume categories,
   > **beleza_saude** leads at about 4.1 stars across ~9,600 reviews.

---

## 📂 Project Structure

```
sql-agent/
├──src
    ├── agent_sql.py          # the LangGraph SQL agent (CLI)
    ├── streamlit_app.py      # interactive Streamlit UI
    ├── load_data.py          # loads the Olist CSVs into MySQL
├── requirements.txt
├── .env                  # secrets (gitignored)
├── .gitignore
└── README.md
```

---

## 🧠 What I Learned Building This

- Designing an agent that **inspects a schema before querying** rather than
  hallucinating table and column names.
- Implementing **human-in-the-loop interrupts** and resumable graph state with
  checkpointers.
- The distinction between **deterministic work** (loading data — a plain script)
  and **agentic work** (analytical reasoning — the LLM).
- Production safety for LLM-over-database systems: read-only guards, graceful SQL
  error handling, and a human approval gate on generated queries.

---

## 🔮 Roadmap

- [ ] A supervisor that routes between this SQL agent and other agents
- [ ] A retrieval (RAG) agent over the review text, fused with SQL results
- [ ] Charts/visualizations of query results
- [ ] Conversation memory across questions

---
