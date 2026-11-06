import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from stockanalysis.stock_research_graph_agent import run_stock_research


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_ENV_VARS = (
    "NEBIUS_API_KEY",
    "NEBIUS_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
)


@st.cache_resource
def get_stock_research_graph():
    from stockanalysis.stock_research_graph_agent import (
        create_stock_research_graph,
    )

    return create_stock_research_graph()


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return str(content)


def main() -> None:
    st.set_page_config(page_title="Stock Research Assistant")
    st.title("Stock Research Assistant")
    st.caption(
        "Research a ticker using stock lessons, recent market data, "
        "financial statements, and technical indicators."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "The research workflow is powered by LangGraph. Price and "
            "financial data come from Yahoo Finance via yfinance; quotes may "
            "be delayed."
        )
        st.caption("This tool provides educational information, not investment advice.")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.thread_id = None
            st.rerun()

    missing_variables = [
        name for name in REQUIRED_ENV_VARS if not os.environ.get(name)
    ]
    if missing_variables:
        st.error(
            "Configure these environment variables in the project .env file "
            f"before using the research graph: {', '.join(missing_variables)}"
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None
    if not st.session_state["messages"]:
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "What would you like to research? "
                    "for example: “Can I buy Google stock?”"
                ),
            }
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_results"):
                with st.expander("Research data and book passages"):
                    for tool_name, result in message["tool_results"]:
                        st.markdown(f"**{tool_name}**")
                        st.code(result, language="json")

    is_clarification = st.session_state.pending_question is not None
    prompt = (
        "Answer the follow-up question..."
        if is_clarification
        else "What would you like to research?"
    )
    user_input = st.chat_input(prompt)
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    if not is_clarification:
        st.session_state.pending_question = user_input.strip()
        st.session_state.thread_id = str(uuid.uuid4())

    try:
        with st.spinner(
            "Continuing the clarification..."
            if is_clarification
            else "Understanding your question and researching..."
        ):
            graph = get_stock_research_graph()
            result = run_stock_research(
                graph,
                question=st.session_state.pending_question,
                thread_id=st.session_state.thread_id,
                clarification=user_input.strip() if is_clarification else None,
            )
    except Exception as exc:
        st.error(f"Stock research failed: {exc}")
        st.exception(exc)
        return

    interruptions = result.get("__interrupt__", [])
    if interruptions:
        interrupt_value = interruptions[0].value
        follow_up = (
            interrupt_value.get("question", "Could you clarify your request?")
            if isinstance(interrupt_value, dict)
            else str(interrupt_value)
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": follow_up}
        )
    else:
        tool_results = [
            (message.name, _message_text(message.content))
            for message in result.get("research_messages", [])
            if getattr(message, "type", None) == "tool"
        ]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": _message_text(result["answer"]),
                "tool_results": tool_results,
            }
        )
        st.session_state.pending_question = None
        st.session_state.thread_id = None

    st.rerun()


if __name__ == "__main__":
    main()
