import ollama

def generate_answer(context,query):
    prompt=f"""
    Use the context to answer the question.
    Context:{context}

    Question:{query}"""

    response=ollama.chat(model="mistral",messages=[{"role":"user","content":prompt}])

    return response["message"]["content"]