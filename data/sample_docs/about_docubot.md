# DocuBot

DocuBot is a RAG-based chatbot API. It lets you ingest documents (plain text or markdown
for now) and then ask questions that get answered strictly using the content of those
documents, with the most relevant source chunks cited in the response.

## How it works

1. Documents are split into overlapping chunks (roughly 800 characters, 150 character overlap).
2. Each chunk is embedded locally using the `all-MiniLM-L6-v2` sentence-transformers model
   and stored in a persistent ChromaDB collection.
3. When a question comes in, DocuBot embeds the question, retrieves the top-k most similar
   chunks, and passes them as context to Claude along with the running conversation history
   for that session.
4. Claude is instructed to answer only from the provided context, and to say so plainly if
   the answer isn't in the documents.

## Why this design

Keeping embeddings local (via sentence-transformers) means you only need one API key
(Anthropic) to run the whole project, which keeps setup friction low for a portfolio demo.
