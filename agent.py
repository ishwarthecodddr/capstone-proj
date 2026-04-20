import os
from typing import List, TypedDict

import numpy as np
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

DOMAIN_NAME = "Code Review Agent"
DOMAIN_DESCRIPTION = (
    "An assistant for developers that reviews code-related questions for bugs, security, "
    "performance, testing, and maintainability."
)

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Code Review Fundamentals and Goals",
        "text": """Code review is a structured process to improve correctness, maintainability, readability, and team knowledge sharing before code reaches production. A high-quality review checks whether the implementation matches requirements, whether failure cases are handled, and whether the code is easy to maintain later. Reviewers should avoid only focusing on style and instead prioritize behavior, risk, and clarity. Effective reviews are specific: point to exact lines, explain why something is risky, and propose an actionable change. Good review comments are objective and constructive, not personal. A practical order for review is: understand intent, verify logic, test edge cases mentally, check security/privacy implications, and finally assess readability and consistency. The outcome should be safer code and stronger team learning, not just approval.""",
    },
    {
        "id": "doc_002",
        "topic": "Bug Patterns in Everyday Code",
        "text": """Common bugs in application code include off-by-one errors, incorrect boundary checks, null/None dereferences, stale state assumptions, race conditions, and incorrect error propagation. Many bugs appear when code works for the happy path but fails for empty input, malformed data, high load, or retries. Developers should verify assumptions at function boundaries: input types, allowed ranges, and missing fields. Another frequent issue is returning partial results after exceptions without clear signaling. Shared mutable state can create non-deterministic behavior in asynchronous code. To reduce defect risk, reviewers look for invariant violations, state transitions, and missing guards. Unit tests should target edge conditions explicitly, not just typical input. If a function has multiple branches, each branch should have at least one targeted test case. Bug-prone logic deserves extra comments or simplification.""",
    },
    {
        "id": "doc_003",
        "topic": "Security Review Basics for Developers",
        "text": """Security-focused review checks whether user-controlled input can change program behavior in dangerous ways. Key risks include injection vulnerabilities, insecure deserialization, path traversal, broken access control, weak authentication handling, and accidental secret leakage. Reviewers should ask: where does untrusted input enter, where is it validated, and where is it used in sensitive operations? Prefer parameterized database queries, strict allowlists, and output encoding based on rendering context. Never log credentials, tokens, or sensitive personal data. File operations should normalize paths and enforce directory boundaries. Error messages should help debugging without exposing internals or secrets. Dependency usage should be current and trusted.""",
    },
    {
        "id": "doc_004",
        "topic": "Performance and Scalability Review Heuristics",
        "text": """Performance review starts by finding repeated expensive work, unnecessary allocations, and unbounded operations. Typical issues include nested loops over large datasets, repeated network calls inside loops, N+1 database queries, and synchronous operations in latency-sensitive paths. Reviewers check algorithmic complexity and memory usage under realistic load, not just tiny samples. Caching is useful when data is stable and invalidation is defined clearly. Batch operations are often better than per-item calls. In APIs, pagination and limits prevent runaway responses. Timeouts, retries with backoff, and circuit breaking improve resilience under dependency failures.""",
    },
    {
        "id": "doc_005",
        "topic": "Readability, Naming, and Maintainability",
        "text": """Maintainable code minimizes cognitive load for future readers. Good naming communicates intent: variables describe meaning, functions describe behavior, and modules reflect responsibilities. Long functions that mix concerns should be split into smaller composable units. Deep nesting often hides logic errors; early returns can make control flow clearer. Comments should explain why decisions were made, not restate obvious syntax. Reviewers should check whether interfaces are stable and whether abstractions are justified. Duplication is a warning sign; duplicated logic should be centralized when practical.""",
    },
    {
        "id": "doc_006",
        "topic": "Error Handling and Observability",
        "text": """Reliable systems treat errors as first-class behavior. Reviewers inspect whether errors are detected, classified, and surfaced with useful context. Silent failures and broad exception swallowing make production issues hard to diagnose. Error messages should include operation context without exposing secrets. Structured logging enables filtering and correlation in monitoring tools. Retries should be bounded and reserved for transient failures; permanent failures should fail fast. Good observability reduces mean time to recovery because teams can identify root causes quickly during incidents.""",
    },
    {
        "id": "doc_007",
        "topic": "Testing Strategy for Review Confidence",
        "text": """Code review is stronger when supported by meaningful tests. A balanced strategy includes unit tests for pure logic, integration tests for component interaction, and end-to-end checks for critical user paths. Reviewers should verify whether tests cover normal cases, edge cases, and failure scenarios. Signs of weak testing include assertions without behavior checks, excessive mocking that hides integration issues, and missing regression tests for previously fixed bugs. Deterministic tests are essential; flaky tests reduce trust in CI.""",
    },
    {
        "id": "doc_008",
        "topic": "API and Contract Review Checklist",
        "text": """For API changes, reviewers validate contract stability, backward compatibility, and clear semantics. Input validation should reject invalid payloads predictably. Response schemas should be explicit and versioning strategy should be defined when breaking changes are possible. HTTP status codes should match outcomes consistently. Idempotency matters for retries in distributed systems, especially on create/update endpoints. Rate limiting, pagination, and timeout behavior should be documented and testable.""",
    },
    {
        "id": "doc_009",
        "topic": "Database and Data Integrity Review",
        "text": """Database-related reviews should confirm correctness, consistency, and operational safety. Queries should use indexes effectively and avoid full scans on large tables unless intentional. Transactions must cover related writes that should succeed or fail together. Reviewers should check isolation assumptions to prevent lost updates and inconsistent reads. Schema migrations need rollback planning and compatibility with rolling deployments. Constraints enforce invariants close to the data and reduce application-level bugs.""",
    },
    {
        "id": "doc_010",
        "topic": "Constructive Review Communication",
        "text": """The quality of communication affects whether review feedback is adopted. Effective comments are specific, respectful, and prioritized by severity. A useful structure is observation, risk, recommendation. Mark blocking issues clearly, and separate them from optional suggestions. Ask questions when intent is unclear rather than assuming mistakes. Acknowledge good design decisions so reviews remain balanced and motivating.""",
    },
]

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2

# Globals initialized in build_agent() and consumed by nodes.
llm = None
embedder = None
collection = None
retriever_mode = "local"
doc_texts = []
doc_topics = []
doc_embeddings = None


class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int


def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:
        msgs = msgs[-6:]
    return {"messages": msgs}


def router_node(state: CapstoneState) -> dict:
    question = state["question"].strip()
    messages = state.get("messages", [])
    recent = "; ".join(f"{m['role']}: {m['content'][:80]}" for m in messages[-3:-1]) or "none"
    q_lower = question.lower()

    memory_triggers = [
        "what did you just say",
        "summarize what you just told me",
        "repeat that",
        "as you said earlier",
        "previous answer",
        "earlier you said",
    ]
    tool_triggers = [
        "latest",
        "official docs",
        "documentation",
        "current version",
        "release notes",
        "new in",
        "breaking changes",
        "cve",
        "security advisory",
    ]

    if any(t in q_lower for t in memory_triggers):
        return {"route": "memory_only"}
    if any(t in q_lower for t in tool_triggers):
        return {"route": "tool"}

    prompt = f"""You are a router for a Code Review Agent used by software developers.

Available options:
- retrieve: for code-review principles and guidance from the knowledge base
- memory_only: when the user asks about earlier conversation context
- tool: when the user asks for latest external info (official docs, versions, release notes, CVEs)

Recent conversation: {recent}
Current question: {question}

Reply with only one word: retrieve / memory_only / tool"""

    decision = llm.invoke(prompt).content.strip().lower()
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"
    return {"route": decision}


def retrieval_node(state: CapstoneState) -> dict:
    q_vec = embedder.encode([state["question"]], normalize_embeddings=True)[0]

    if retriever_mode == "chroma" and collection is not None:
        results = collection.query(query_embeddings=[q_vec.tolist()], n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
    else:
        sims = np.dot(doc_embeddings, q_vec)
        top_idx = np.argsort(sims)[::-1][:3]
        chunks = [doc_texts[i] for i in top_idx]
        topics = [doc_topics[i] for i in top_idx]

    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    question = state["question"]
    try:
        search_query = f"{question} official documentation best practices"
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))

        if not results:
            return {"tool_result": "No web results found for this query."}

        lines = []
        for r in results[:3]:
            title = r.get("title", "No title")
            link = r.get("href") or r.get("url", "")
            body = (r.get("body", "") or "").replace("\n", " ").strip()
            lines.append(f"- {title}\n  {body[:220]}\n  Source: {link}")
        return {"tool_result": "Top web findings:\n" + "\n\n".join(lines)}
    except Exception as e:
        return {"tool_result": f"Web search error: {e}"}


def answer_node(state: CapstoneState) -> dict:
    question = state["question"]
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    if context:
        system_content = f"""You are a Code Review Agent helping software developers.

Use ONLY the provided context.
If context is insufficient, say exactly:
I don't have that information in my knowledge base.

When possible, format answer as:
1) Issue/Risk
2) Why it matters
3) Suggested fix

Be precise and actionable. Do not invent facts.

{context}"""
    else:
        system_content = "You are a Code Review Agent. Answer only from conversation history."

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Previous answer failed quality check. Stay strictly grounded in context."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}


def eval_node(state: CapstoneState) -> dict:
    answer = state.get("answer", "")
    context = state.get("retrieved", "")[:500]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":
        return "tool"
    if route == "memory_only":
        return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    score = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


def build_agent():
    llm_local = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder_local = SentenceTransformer("all-MiniLM-L6-v2")

    texts_local = [d["text"] for d in DOCUMENTS]
    topics_local = [d["topic"] for d in DOCUMENTS]
    embeddings_local = np.array(
        embedder_local.encode(texts_local, normalize_embeddings=True),
        dtype=np.float32,
    )

    collection_local = None
    retriever_mode_local = "local"
    try:
        import chromadb

        client_local = chromadb.Client()
        try:
            client_local.delete_collection("capstone_kb")
        except Exception:
            pass
        collection_local = client_local.create_collection("capstone_kb")
        collection_local.add(
            documents=texts_local,
            embeddings=embeddings_local.tolist(),
            ids=[d["id"] for d in DOCUMENTS],
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
        )
        retriever_mode_local = "chroma"
    except Exception:
        # Cloud environments can fail importing chromadb due to protobuf/opentelemetry constraints.
        retriever_mode_local = "local"

    # Bind globals consumed by node functions.
    global llm, embedder, collection, retriever_mode, doc_texts, doc_topics, doc_embeddings
    llm = llm_local
    embedder = embedder_local
    collection = collection_local
    retriever_mode = retriever_mode_local
    doc_texts = texts_local
    doc_topics = topics_local
    doc_embeddings = embeddings_local

    graph = StateGraph(CapstoneState)
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app, len(texts_local), retriever_mode_local
