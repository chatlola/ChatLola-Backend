import joblib
import json

with open("chatlola/knowledge_base.json", "r", encoding="utf-8") as file:
    chatlola_data = json.load(file)

def intent_recognition(query):
    intent_model = joblib.load("model/intent_model.pkl")
    intent_utterance_vectorizer = joblib.load("model/intent_tfidf_vectorizer.pkl")

    vectorized_query = intent_utterance_vectorizer.transform([query])[0]
    
    intent = intent_model.predict(vectorized_query)[0]

    return intent

def conversation_management(query, intent):
    intent_data = chatlola_data[intent]

    # skip empty intents for now
    if not intent_data:
        return f"Can't find response for query with intent: {intent}", [["define_concept", "scam"], [["define_concept", "digital safety"]]]

    for tag_name, tag_data in intent_data.items():
        for keyword in tag_data["keywords"]:
            if keyword in query:
                return tag_data["response"], tag_data["related"]
    
    # would add trying to infer appropriate response from context
    # for now just skip
    return f"Can't find response for query with intent: {intent}", [["define_concept", "scam"], [["define_concept", "digital safety"]]]
