# Code Review Agent - Agentic AI Capstone Project

This project is an **Agentic AI Code Review Assistant** designed to help software developers by reviewing code-related questions for bugs, security, performance, testing, and maintainability.

Built as an advanced workflow-driven AI application, it utilizes an intelligent routing system to deduce whether to rely on conversation memory, retrieve best practices from an embedded knowledge base, or search the live web for the latest documentation and vulnerabilities.

## 🚀 Key Features

* **Agentic Routing**: Intelligently decides between memory, internal knowledge retrieval, or external web search based on the user's query context.
* **Knowledge Retrieval (RAG)**: Uses a local, domain-specific knowledge base containing code review fundamentals, security review basics, and performance heuristics. Embeddings are handled via `SentenceTransformers` and vector storage via `ChromaDB` (with lightweight local array fallback).
* **Live Web Tooling**: Has the ability to query the web (via DuckDuckGo Search) when a user specifically asks for the latest official docs, newer versions, or CVE details.
* **Self-Evaluation & Anti-Hallucination**: The agent rates its own answers for *faithfulness* against the retrieved context. If an answer fails the quality check, the agent automatically retries the generation.
* **Interactive UI**: Clean, interactive chat interface built with **Streamlit**, featuring session tracking and conversation memory.
* **Robust Deployment Support**: Ready for deployment on platforms like Streamlit Cloud, seamlessly handling environment variables and Streamlit secrets.

## 🛠️ Architecture & Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Agent Framework**: [LangGraph](https://github.com/langchain-ai/langgraph) & [LangChain](https://langchain.com/)
* **LLM Provider**: [Groq](https://groq.com/) (using the `llama-3.3-70b-versatile` model for high-speed, accurate reasoning)
* **Embeddings**: `SentenceTransformer` (`all-MiniLM-L6-v2`)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/) (with an in-memory fallback mechanism)
* **Search Tool**: `ddgs` (DuckDuckGo Search)

## 📁 Project Structure

```text
.
├── agent.py                 # Core LangGraph state graph, embedding, retrieval, and evaluation nodes
├── capstone_streamlit.py    # Streamlit application UI and chat session state management
├── day13_capstone.ipynb     # Jupyter Notebook used for initial prototyping and experimentation
├── requirements.txt         # Python dependencies
├── runtime.txt              # Specifies the Python runtime version for cloud deployments
└── README.md                # Project documentation
```

## ⚙️ Prerequisites

* Python 3.10+ (Check your `runtime.txt` or environment)
* A valid **Groq API Key**. You can get one for free at [console.groq.com](https://console.groq.com/).

## 💻 Setup & Installation

1. **Clone the repository** and navigate to the project directory.

2. **Create a virtual environment**:
   ```bash
   python -m venv myvenv
   ```

3. **Activate the virtual environment**:
   * **Windows**: `myvenv\Scripts\activate`
   * **Mac/Linux**: `source myvenv/bin/activate`

4. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure your Environment Variables**:
   * Create a `.env` file in the root directory.
   * Add your Groq API key:
     ```env
     GROQ_API_KEY=your_api_key_here
     ```

## ▶️ Running the Application

To start the Streamlit application locally, run the following command:

```bash
streamlit run capstone_streamlit.py
```

The application will launch in your default web browser, usually at `http://localhost:8501`.

## ☁️ Deployment Notes (e.g., Streamlit Community Cloud)

When deploying to a cloud service like Streamlit Community Cloud, the application will look for API keys in the environment's secrets configuration.

1. Ensure `GROQ_API_KEY` is added to the application secrets (in Streamlit Cloud: *Settings* -> *Secrets*).
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```
2. The application is configured to read from `st.secrets` during deployment to prevent startup failures where local `.env` files are not available.
