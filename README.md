# Stock Analysis Book Chat

A small Streamlit proof of concept for asking questions about the stock-education
book. It retrieves relevant passages from Pinecone and sends them, along with
the question, to a Nebius-hosted chat model. This is an educational Q&A demo,
not a source of live prices or personalized investment advice.

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
   uv run streamlit run app.py
   ```

The app keeps the conversation in Streamlit session state and displays the
retrieved book titles and page numbers for each answer.
