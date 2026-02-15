def hybrid_rank(text_results,image_results,table_results):
    combined=[]

    for score,content in text_results:
        combined.append(("text",score,content))

    for score,content in image_results:
        combined.append(("image",score,content))

    for score,content in table_results:
        combined.append(("table",score,content))

    combined.sort(key=lambda x:x[1])

    return combined[:5]