import joblib
import json
from flask import jsonify

with open("chatlola/knowledge_base.json", "r", encoding="utf-8") as file:
    chatlola_data = json.load(file)

with open("chatlola/context.json", "r", encoding="utf-8") as file:
    context_data = json.load(file)

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

def conversation_management(query, intent, confusion_label, prev_intent, prev_topic, context):
    intent_data = chatlola_data[intent]

    # skip empty intents for now
    if not intent_data:
        return { "response": None, "confusion_label": confusion_label }
    
    for tag_name, tag_data in intent_data.items():
        for key1 in tag_data["keywords"]["first"]:
            if key1 in query:
                if "second" not in tag_data["keywords"]:
                    return return_response_data(confusion_label, prev_intent, prev_topic, intent, tag_data, tag_name)
                else:
                    for key2 in tag_data["keywords"]["second"]:
                        if key2 in query:
                            return return_response_data(confusion_label, prev_intent, prev_topic, intent, tag_data, tag_name)

    #if no specific response is found
    return no_keys_response(intent, context, prev_intent, prev_topic, confusion_label)

    #return no response
    return return_response_data(confusion_label, prev_intent, prev_topic, intent, { "response": None }, "")

def return_response_data(confusion_label, prev_intent, prev_topic, intent, tag_data, tag_name):
    #will also handle here if confusion is detected
    
    if confusion_label:
        # if tag_data isn't empty
        if tag_data["response"] != None:

            #if intent is emotional_support then return as is
            if intent == "emotional_support":
                return jsonify({
                    **{k: v for k, v in tag_data.items() if k != "keywords" and k != "confused_response"},
                    "topic": tag_name,
                    "intent": intent,
                    "confusion_label": confusion_label #only for testing
                })

            # output tag_data but with the confused response
            return jsonify({
                **{k: v for k, v in tag_data.items() if k != "keywords" and k != "confused_response" and k != "response"},
                "topic": tag_name,
                "intent": intent,
                "response": tag_data["confused_response"],
                "confusion_label": confusion_label #only for testing
            })

        #if confused and prev data are available, then send the response data of the previous topic but for the confused response
        if (prev_intent != "" and prev_intent != None) and (prev_topic != "" and prev_topic != None) and (prev_intent != None and prev_topic != "clarify") :
            print("got to here")
            response_data = chatlola_data[prev_intent][prev_topic]

            return jsonify({
                **{k: v for k, v in response_data.items() if k != "keywords" and k != "confused_response" and k != "response"},
                "topic": "clarify",
                "intent": None,
                "response": "Mukhang medyo nakakalito 😅. " + response_data["confused_response"],
                "confusion_label": confusion_label #only for testing
            })
        
        return jsonify({
            "topic": "clarify",
            "intent": None,
            "response": "Maaari mo bang linawin kung anong bahagi o topic ang gusto mong maintindihan?",
            "confusion_label": confusion_label,  # only for testing
            "context": "general"
        })

    #if not confused then just output what was sent to this func
    return jsonify({
        **{k: v for k, v in tag_data.items() if k != "keywords" and k != "confused_response"},
        "topic": tag_name,
        "intent": intent,
        "confusion_label": confusion_label #only for testing
    })

def no_keys_response(intent, context, prev_intent, prev_topic, confusion_label): 
    if intent == "explain_concept" or (context is None or context == ""):
        return { 
            "response": None, 
            "topic": "clarify",
            "intent": intent,
            "context": context
        }

    print(context)
    current_context_data = context_data[context]

    #find the response that belongs to context and have the same intent as the predicted intent
    for topic_name, topic_data in current_context_data.items():
        if topic_data["intent"] == intent:

            response_data = chatlola_data[intent][topic_name]

            # if previous intent and topic matches the response found, then output the confused response 
            if (prev_topic == topic_name and prev_intent == intent) or confusion_label == 1:
                return {
                    "response": response_data["confused_response"],
                    "intent": intent,
                    "topic": topic_name,
                    "context": response_data["context"]
                }
            else:
                return {
                    "response": response_data["response"],
                    "intent": intent,
                    "topic": topic_name,
                    "context": response_data["context"]
                }

def clarify_response(query):
    #maybe there is an easier way of doing this
    for intent_name, intent_data in chatlola_data.items():
        for tag_name, tag_data in intent_data.items():
            for key1 in tag_data["keywords"]["first"]:
                if key1 in query:
                    if "second" not in tag_data["keywords"]:
                        return jsonify({
                            "response": tag_data["response"],
                            "intent": intent_name,
                            "topic": tag_name,
                            "context": tag_data["context"]
                        })
                    else:
                        for key2 in tag_data["keywords"]["second"]:
                            if key2 in query:
                                return jsonify({
                                    "response": tag_data["response"],
                                    "intent": intent_name,
                                    "topic": tag_name,
                                    "context": tag_data["context"]
                                })

    return { 
        "response": None, 
        "topic": "clarify",
        "intent": None,
        "context": "general"
        }