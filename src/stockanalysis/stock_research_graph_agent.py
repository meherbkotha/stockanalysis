import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pinecone import Pinecone
from pydantic import BaseModel, Field

from stockanalysis.utilities import (
    get_company_financials as fetch_company_financials,
    get_stock_history as fetch_stock_history,
    get_technical_indicators as fetch_technical_indicators,
)
from stockanalysis.yahoo_finance import get_latest_stock_price


class StockState(TypedDict, total=False):
    question: str
    symbol: str
    horizon: str
    clarifications: list[str]
    missing_field: str
    research_messages: list
    answer: str


class ResearchContext(BaseModel):
    symbol: str | None = Field(
        description=(
            "The stock ticker identified in the conversation. Resolve a clearly "
            "named company to its ticker; use GOOGL for Google unless GOOG was "
            "explicitly requested. Leave null if unclear."
        )
    )
    horizon: str | None = Field(
        description=(
            "The investment timeframe explicitly stated by the user. Do not "
            "guess; leave null if the user did not state one."
        )
    )


SYSTEM_PROMPT = (
    "You help users research stocks using the available tools. For current "
    "prices, call get_latest_price. For historical prices, call "
    "get_stock_history. For RSI and moving averages, call "
    "get_technical_indicators. For financial statements, call "
    "get_company_financials. For trading concepts, search stock lessons and "
    "cite available sources and page numbers. For broad buy/sell questions, "
    "fetch the latest price, technical indicators, and financial statements "
    "before answering. Use the confirmed ticker and investment horizon. "
    "Include timestamps, statement periods, and sources. Explain tool "
    "failures and missing data specifically. Do not invent prices, "
    "indicators, news, or forecasts. Do not claim tools are unavailable when "
    "they are registered. Web browsing is not provided by these tools. This "
    "is educational information, not personalized investment advice."
)


def create_stock_research_graph():
    """Create the stock-research LangGraph and its configured tool-using agent."""
    load_dotenv()

    llm = ChatOpenAI(
        model=os.environ["NEBIUS_MODEL"],
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url="https://api.tokenfactory.nebius.com/v1/",
        temperature=0,
        use_responses_api=False,
    )

    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pinecone = Pinecone(api_key=pinecone_api_key)
    index = pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=pinecone_api_key,
        query_params={"input_type": "query", "dimension": 512},
        document_params={"input_type": "passage", "dimension": 512},
    )
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace="stock-analysis-nsp",
    )

    @tool
    def search_stock_lessons(query: str) -> str:
        """Search the stock-analysis lessons for relevant concepts."""
        documents = vector_store.similarity_search(query, k=4)
        if not documents:
            return "No relevant passages found."

        return json.dumps(
            [
                {
                    "content": document.page_content,
                    "source": document.metadata.get("source"),
                    "title": document.metadata.get("title"),
                    "page_number": document.metadata.get("page_number"),
                }
                for document in documents
            ],
            ensure_ascii=False,
        )

    @tool
    def get_latest_price(symbol: str) -> dict:
        """Get the latest available stock price bar for a ticker."""
        return get_latest_stock_price(symbol)

    @tool
    def get_company_financials(symbol: str, frequency: str = "quarterly") -> dict:
        """Get up to four reported income, balance-sheet, and cash-flow periods."""
        return fetch_company_financials(symbol, frequency)

    @tool
    def get_stock_history(symbol: str, period: str = "3mo") -> dict:
        """Get adjusted daily price history for 1mo, 3mo, 6mo, 1y, or 2y."""
        return fetch_stock_history(symbol, period)

    @tool
    def get_technical_indicators(symbol: str) -> dict:
        """Calculate 20/50/200-day SMAs and 14-day RSI from adjusted prices."""
        return fetch_technical_indicators(symbol)

    research_agent = create_agent(
        model=llm,
        tools=[
            get_latest_price,
            search_stock_lessons,
            get_stock_history,
            get_technical_indicators,
            get_company_financials,
        ],
        system_prompt=SYSTEM_PROMPT,
    )

    context_extractor = llm.with_structured_output(ResearchContext)
    context_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only the stock ticker and investment horizon from the "
                "conversation. The original request and follow-up answers are "
                "from the user. A company name may be resolved to a ticker "
                "only when unambiguous. Never assume an investment horizon. "
                "If context is still missing, return null for that field.",
            ),
            (
                "human",
                "Original request:\n{question}\n\n"
                "Follow-up answers:\n{clarifications}\n\n"
                "Already identified ticker: {symbol}\n"
                "Already identified horizon: {horizon}",
            ),
        ]
    )
    context_chain = context_prompt | context_extractor

    def extract_context(state: StockState) -> dict:
        context = context_chain.invoke(
            {
                "question": state["question"],
                "clarifications": "\n".join(
                    state.get("clarifications", [])
                ) or "(none)",
                "symbol": state.get("symbol") or "(not identified)",
                "horizon": state.get("horizon") or "(not identified)",
            }
        )
        symbol = (context.symbol or state.get("symbol") or "").strip().upper()
        horizon = (context.horizon or state.get("horizon") or "").strip()
        missing_field = "symbol" if not symbol else "horizon" if not horizon else ""
        return {
            "symbol": symbol,
            "horizon": horizon,
            "missing_field": missing_field,
        }

    def route_after_context(state: StockState) -> str:
        return "clarify" if state.get("missing_field") else "research"

    def clarify(state: StockState) -> dict[str, list[str]]:
        if state["missing_field"] == "symbol":
            question = (
                "Which company or ticker symbol should I research? "
                "For example, Google (GOOGL)."
            )
        else:
            question = (
                "What investment timeframe should I use—for example, "
                "short term or 3 to 5 years?"
            )
        response = interrupt({"question": question})
        clarifications = list(state.get("clarifications", []))
        clarifications.append(str(response).strip())
        return {"clarifications": clarifications}

    def research(state: StockState) -> dict:
        result = research_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Research question: {state['question']}\n"
                            f"Confirmed ticker: {state['symbol']}\n"
                            f"Investment horizon: {state['horizon']}"
                        ),
                    }
                ]
            }
        )
        messages = result["messages"]
        if not messages:
            raise RuntimeError("The research agent returned no messages.")
        return {
            "research_messages": messages,
            "answer": messages[-1].content,
        }

    builder = StateGraph(StockState)
    builder.add_node("extract_context", extract_context)
    builder.add_node("clarify", clarify)
    builder.add_node("research", research)
    builder.add_edge(START, "extract_context")
    builder.add_conditional_edges(
        "extract_context",
        route_after_context,
        {"clarify": "clarify", "research": "research"},
    )
    builder.add_edge("clarify", "extract_context")
    builder.add_edge("research", END)
    return builder.compile(checkpointer=InMemorySaver())


def run_stock_research(
    graph,
    *,
    question: str,
    thread_id: str,
    clarification: str | None = None,
) -> dict:
    """Start research or resume it with an answer to the graph's follow-up."""
    config = {"configurable": {"thread_id": thread_id}}
    if clarification is None:
        return graph.invoke({"question": question}, config=config)
    return graph.invoke(
        Command(resume=clarification),
        config=config,
    )
