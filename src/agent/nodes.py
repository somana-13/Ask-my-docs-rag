from __future__ import annotations
import re
from src.tools.retrieval_tool import retrieve_documents
def clean_evidence_snippet(text: str, max_sentences: int = 2) -> str:
    """
    Convert raw retrieved markdown into cleaner sentences for answer generation.
    This avoids artificial ellipses and markdown bullets that confuse NLI scoring.
    """
    text = text.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    selected = []
    bad_endings = ("...", "following...", "a.k.a.", "e.g.", "i.e.")

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        if sentence.lower().endswith(bad_endings):
            continue

        selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return " ".join(selected).strip()

def router_node(state: AgentState) -> AgentState:
    """
    Route the query to the right agent path.

    Routes:
    - simple_search: direct factual/how-to questions
    - decompose_multihop: questions that combine multiple concepts
    - clarify: vague or underspecified queries
    """
    query = state["query"].strip()
    query_lower = query.lower()
    query_normalized = re.sub(r"[^\w\s]", "", query_lower).strip()

    vague_queries = {
        "help",
        "explain",
        "tell me more",
        "what about it",
        "how does it work",
        "what is this",
    }

    unsupported_topic_markers = [
        # intentionally removed for CI demo: "aws",
        # intentionally removed for CI demo: "lambda",
        # intentionally removed for CI demo: "kubernetes",
        "docker",
        # intentionally removed for CI demo: "deploy",
        "deployment",
        "terraform",
        "ec2",
        "s3",
        "cloud",
        "bert",
        "fine",
        "finetune",
        "fine-tune",
        "fine-tuning",
        "train",
        "training",
        "model",
    ]

    if any(marker in query_normalized.split() for marker in unsupported_topic_markers):
        return {
            **state,
            "route": "clarify",
            "needs_clarification": True,
            "clarification_question": (
                "I could not find this deployment/cloud topic in the indexed HTTPX documentation. "
                "Can you ask about authentication, timeouts, SSL, redirects, or another indexed HTTPX topic?"
            ),
        }

    if not query or query_normalized in vague_queries or len(query_normalized.split()) < 4:
        return {
            **state,
            "route": "clarify",
            "needs_clarification": True,
            "clarification_question": (
                "Can you clarify what specific HTTPX topic you want to ask about?"
            ),
        }

    multi_hop_patterns = [
        r"\band\b",
        r"\bboth\b",
        r"\bcompare\b",
        r"difference between",
        r"\btogether\b",
        r"\bcombined\b",
        r"\bmultiple\b",
        r"authentication.*timeout",
        r"timeout.*authentication",
        r"ssl.*timeout",
        r"timeout.*ssl",
        r"redirect.*authentication",
        r"authentication.*redirect",
    ]

    if any(re.search(pattern, query_lower) for pattern in multi_hop_patterns):
        return {
            **state,
            "route": "decompose_multihop",
            "needs_clarification": False,
        }

    return {
        **state,
        "route": "simple_search",
        "needs_clarification": False,
    }


def clarify_node(state: AgentState) -> AgentState:
    """
    Return a clarification question instead of retrieving when the query is vague.
    """
    clarification_question = state.get(
        "clarification_question",
        "Can you clarify what specific topic you want to ask about?",
    )

    return {
        **state,
        "answer": clarification_question,
        "citations": [],
        "sources": [],
        "retrieved_chunks": [],
    }


def decompose_node(state: AgentState) -> AgentState:
    """
    Break a multi-hop query into smaller retrieval-friendly sub-questions.

    This deterministic decomposer maps HTTPX topic keywords to focused
    sub-questions so multi-hop retrieval covers each required source.
    """
    query = state["query"].strip()
    query_lower = query.lower()

    topic_rules = [
        (
            ["authentication", "auth", "basic authentication", "digest authentication"],
            "How do I configure authentication in HTTPX?",
        ),
        (
            ["timeout", "timeouts"],
            "How do I configure timeouts in HTTPX?",
        ),
        (
            ["ssl", "certificate", "certificates", "certificate verification"],
            "How do I configure SSL certificates in HTTPX?",
        ),
        (
            ["redirect", "redirects"],
            "How do redirects work in HTTPX?",
        ),
        (
            ["client", "clients", "connection settings", "sync clients"],
            "How do I use clients in HTTPX?",
        ),
        (
            ["async", "async requests", "async clients"],
            "How do I use async support in HTTPX?",
        ),
        (
            ["proxy", "proxies"],
            "How do proxies work in HTTPX?",
        ),
        (
            ["transport", "transports"],
            "How do transports work in HTTPX?",
        ),
        (
            ["exception", "exceptions", "error", "errors"],
            "How do exceptions work in HTTPX?",
        ),
        (
            ["environment variable", "environment variables", "ssl_cert_file", "ssl_cert_dir"],
            "How do environment variables work in HTTPX?",
        ),
    ]

    sub_questions: list[str] = []
    seen = set()

    for keywords, sub_question in topic_rules:
        if any(keyword in query_lower for keyword in keywords):
            if sub_question not in seen:
                sub_questions.append(sub_question)
                seen.add(sub_question)

    # Fallback: split explicit conjunction/comparison queries.
    if not sub_questions:
        parts = re.split(
            r"\band\b|\bboth\b|\bcompare\b|\bdifference between\b|,",
            query,
            flags=re.IGNORECASE,
        )
        sub_questions = [part.strip() for part in parts if len(part.strip().split()) >= 3]

    if len(sub_questions) < 2:
        sub_questions = [query]

    return {
        **state,
        "sub_questions": sub_questions,
    }


def retrieve_subquestions_node(state: AgentState) -> AgentState:
    """
    Run retrieval separately for each decomposed sub-question.
    """
    sub_questions = state.get("sub_questions", [])
    subquestion_results = []
    combined_chunks = []
    combined_sources = []
    seen_chunk_ids = set()
    seen_sources = set()

    for sub_question in sub_questions:
        result = retrieve_documents(sub_question)
        subquestion_results.append(
            {
                "sub_question": sub_question,
                "retrieval_result": result,
            }
        )

        for chunk in result.get("evidence", []):
            chunk_id = chunk.get("chunk_id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            combined_chunks.append(chunk)

        for source in result.get("sources", []):
            if source in seen_sources:
                continue
            seen_sources.add(source)
            combined_sources.append(source)

    return {
        **state,
        "subquestion_results": subquestion_results,
        "retrieved_chunks": combined_chunks,
        "sources": combined_sources,
        "retrieval_result": {
            "query": state.get("query", ""),
            "has_evidence": bool(combined_chunks),
            "evidence": combined_chunks,
            "sources": combined_sources,
            "subquestion_results": subquestion_results,
        },
    }


def synthesis_node(state: AgentState) -> AgentState:
    """
    Synthesize a final answer from evidence retrieved for multiple sub-questions.
    """
    subquestion_results = state.get("subquestion_results", [])

    if not subquestion_results:
        return {
            **state,
            "answer": "I could not decompose the query into supported sub-questions.",
            "citations": [],
        }

    answer_sections = []
    citations = []

    for item in subquestion_results:
        sub_question = item.get("sub_question", "")
        retrieval_result = item.get("retrieval_result", {})
        evidence = retrieval_result.get("evidence", [])

        if not evidence:
            answer_sections.append(
                f"For sub-question '{sub_question}', I could not find strong supporting evidence."
            )
            continue

        top_chunk = evidence[0]
        source_name = top_chunk.get("source_name", "unknown")
        section_title = top_chunk.get("section_title", "unknown")
        raw_text = top_chunk.get("raw_text", "")
        snippet = clean_evidence_snippet(raw_text)

        answer_sections.append(
            f"For '{sub_question}', the relevant documentation says: {snippet}"
        )
        citations.append(f"{source_name} | {section_title}")

    answer = (
        "This question requires evidence from multiple parts of the documentation.\n\n"
        + "\n\n".join(answer_sections)
    )

    return {
        **state,
        "answer": answer,
        "citations": citations,
    }


def simple_search_node(state: AgentState) -> AgentState:
    """
    Run the existing retrieval pipeline as an agent node.

    This is the first bridge between your RAG system and LangGraph.
    """
    query = state["query"]
    retrieval_result = retrieve_documents(query)

    return {
        **state,
        "route": state.get("route", "simple_search"),
        "retrieval_result": retrieval_result,
        "retrieved_chunks": retrieval_result.get("evidence", []),
        "sources": retrieval_result.get("sources", []),
    }


def answer_node(state: AgentState) -> AgentState:
    """
    Create a simple evidence-based answer.

    For now, this is extractive/template-based. Later, we can replace this
    with an LLM generation step.
    """
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            **state,
            "answer": (
                "I could not find strong enough supporting evidence in the indexed documents."
            ),
            "citations": [],
        }

    evidence_lines = []
    citations = []

    for chunk in chunks[:3]:
        source_name = chunk.get("source_name", "unknown")
        section_title = chunk.get("section_title", "unknown")
        raw_text = chunk.get("raw_text", "")
        snippet = clean_evidence_snippet(raw_text)

        if snippet:
            evidence_lines.append(f"- {snippet}")
        citations.append(f"{source_name} | {section_title}")

    answer = (
        "Based on the retrieved documentation, here is the most relevant evidence:\n\n"
        + "\n".join(evidence_lines)
    )

    return {
        **state,
        "answer": answer,
        "citations": citations,
    }
