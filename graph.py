from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from rag import answer_question, direct_answer


class State(TypedDict):
    question: str
    intent: str
    answer: str


def classify_intent(state: State) -> dict:
    """Route questions into 'policy_question' or 'general_question'.

    Covers all policy topics in docs/ (doc_01 through doc_08):
    delivery, refunds, memberships, tracking, cancellation, damaged items,
    gift cards, and customer support hours.
    """
    question = state["question"].lower()

    policy_keywords = [
        # doc_01: Delivery Policy & Charges
        "delivery", "deliver", "delayed", "delay", "charges", "fee", "shipping", "timing",
        # doc_02: Returns & Refunds
        "refund", "return", "incorrect", "wrong item", "payment",
        # doc_03: Membership Tiers
        "membership", "tier", "benefit", "pass", "savings",
        # doc_04: Order Tracking
        "track", "tracking", "status", "dispatch", "where is", "order",
        # doc_05: Order Cancellation Policy
        "cancel", "cancellation",
        # doc_06: Damaged or Missing Items
        "damaged", "damage", "broken", "missing", "leak", "spoiled", "rotten", "stale", "replacement",
        # doc_07: Gift Cards
        "gift card", "voucher", "coupon", "promo", "redeem", "redemption",
        # doc_08: Customer Support Hours & Escalation
        "support", "hours", "contact", "agent", "escalat", "customer care", "help", "reach",
    ]

    if any(word in question for word in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: State) -> dict:
    """RAG path — fetch from vector store, then LLM."""
    return {"answer": answer_question(state["question"])}


def direct_answer_node(state: State) -> dict:
    """Direct LLM path — no retrieval (for general questions)."""
    return {"answer": direct_answer(state["question"])}


def route_question(state: State) -> str:
    """Conditional edge: decide which node to call next."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


# Build the LangGraph pipeline
graph_builder = StateGraph(State)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer_node)

graph_builder.add_edge(START, "classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

graph = graph_builder.compile()