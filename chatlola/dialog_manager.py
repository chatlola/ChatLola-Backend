import joblib
import json

with open("chatlola/knowledge_base.json", "r", encoding="utf-8") as file:
    chatlola_data = json.load(file)

def intent_recognition(query):
    intent_model = joblib.load("models/intent/intent_model.pkl")
    intent_utterance_vectorizer = joblib.load("models/intent/intent_tfidf_vectorizer.pkl")

    vectorized_query = intent_utterance_vectorizer.transform([query])[0]
    
    intent = intent_model.predict(vectorized_query)[0]

    return intent

def confusion_detection(query):
    confusion_detection_model = joblib.load("models/confusion/confusion_model.pkl")
    user_utterance_vectorizer = joblib.load("models/confusion/confusion_tfidf_vectorizer.pkl")

    vectorized_query = user_utterance_vectorizer.transform([query])[0]
    
    confusion_label = confusion_detection_model.predict(vectorized_query)[0]

    return confusion_label

def conversation_management(query, intent, confusion_label):
    intent_data = chatlola_data[intent]

    # skip empty intents for now
    if not intent_data:
        return { "response": None }, ""
    
    for tag_name, tag_data in intent_data.items():
        for key1 in tag_data["keywords"]["first"]:
            if key1 in query:
                if "second" not in tag_data["keywords"]:
                    return return_response_data(confusion_label, intent_data, tag_data, tag_name)
                else:
                    for key2 in tag_data["keywords"]["second"]:
                        if key2 in query:
                            return return_response_data(confusion_label, intent_data, tag_data, tag_name)

    return { "response": None }, ""

def return_response_data(confusion_label, intent_data, tag_data, tag_name):
    #will also handle here if confusion is detected
    
    return tag_data, tag_name