def route_query(query):
    keywords=["diagram","image","graph","figure","plot"]


    for word in keywords:
        if word in query.lower():
            return "image"
        
    return "text"