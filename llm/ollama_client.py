import ollama

def generate_answer(context,query):
    prompt = f"""
        Answer using ONLY the provided context.
        If not found, say 'Not found in documents.'

        Context:
        {context}

        Return:
        - Answer
        - Bullet list of sources
        """
    response=ollama.chat(model="mistral",messages=[{"role":"user","content":prompt}])

    return response["message"]["content"]