import ollama

def embed_text(chunks):
    embeddings = []

    for chunk in chunks:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(response["embedding"])

    return embeddings
