<<<<<<< HEAD
# Stock Research Assistant
=======
# Stock Analysis Agent
>>>>>>> 17d6c55e2c772af08a388c3f9524c8c7bbdda1f6

A Streamlit proof of concept for stock research. The agent can search the
stock-analysis book and call tools for recent prices, historical price data,
financial statements, and technical indicators. It uses a Nebius-hosted chat
model and Pinecone for book retrieval. Market data is provided by Yahoo Finance
via `yfinance`; quotes may be delayed. This is not personalized investment
advice.

## Run locally

1. Install the project dependencies:

   ```powershell
   uv sync
   ```

2. Create a `.env` file in the project root and configure:

   ```text
   NEBIUS_API_KEY=your-nebius-api-key
   NEBIUS_MODEL=your-nebius-model-name
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-pinecone-index-name
   ```

   Keep API keys private; `.env` is ignored by Git.

3. Ensure the Pinecone index has been populated with the book passages in the
   `stock-analysis-nsp` namespace, using 512-dimensional
   `llama-text-embed-v2` embeddings.

4. Start the chat UI:

   ```powershell
   uv run streamlit run app_stock_research_agent.py
   ```

The app keeps the conversation in Streamlit session state and shows tool
outputs and retrieved book passages under each answer.
