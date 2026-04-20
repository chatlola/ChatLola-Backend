import joblib
import json
from flask import jsonify


# --- Load data ---
with open("chatlola/knowledge_base.json", encoding="utf-8") as f:
    chatlola_data = json.load(f)

with open("chatlola/context.json", encoding="utf-8") as f:
    context_data = json.load(f)


# --- Load models ---
intent_model = joblib.load("models/intent/intent_model.pkl")
intent_vectorizer = joblib.load("models/intent/intent_tfidf_vectorizer.pkl")

confusion_model = joblib.load("models/confusion/confusion_model.pkl")
confusion_vectorizer = joblib.load("models/confusion/confusion_tfidf_vectorizer.pkl")


# --- Helpers ---
def vectorize(text, vectorizer):
    return vectorizer.transform([text])

def match_keywords(query, keywords):
    if not any(k in query for k in keywords["first"]):
        return False

    if "second" not in keywords:
        return True

    return any(k in query for k in keywords["second"])

def build_response(tag_data, intent, topic, confusion_label, confused=False):
    if not tag_data:
        return jsonify({
            "response": None,
            "intent": intent,
            "topic": topic,
            "confusion_label": confusion_label
        })

    response_text = (
        tag_data.get("confused_response")
        if confused else tag_data.get("response")
    )

    topic = (
        topic + "_confused"
        if confused else topic
    )

    return jsonify({
        **{k: v for k, v in tag_data.items()
           if k not in ("keywords", "confused_response", "response")},
        "response": response_text,
        "intent": intent,
        "topic": topic,
        "confusion_label": confusion_label
    })


# --- Core Models ---
def intent_recognition(query):
    vec = vectorize(query, intent_vectorizer)
    return intent_model.predict(vec)[0]

def confusion_detection(query):
    vec = vectorize(query, confusion_vectorizer)
    return confusion_model.predict(vec)[0]


# --- Main Logic ---
def conversation_management(query, intent, confusion, prev_intent, prev_topic, context):
    intent_data = chatlola_data.get(intent)

    if not intent_data:
        return build_response(None, intent, None, confusion)

    # 1. Try keyword match
    for topic, data in intent_data.items():
        if match_keywords(query, data["keywords"]):
            return handle_confusion(data, intent, topic, confusion, prev_intent, prev_topic)

    # 2. Fallback logic
    if intent == "explain_concept":
        if confusion:
            return handle_confusion({ "response": None }, intent, "", confusion, prev_intent, prev_topic)
        
        return jsonify({
            "response": None,
            "intent": intent,
            "topic": "clarify",
            "context": context
        })

    # 3. No context → clarify
    if not context:
        return jsonify({
            "response": None,
            "intent": intent,
            "topic": "clarify",
            "context": context
        })

    # 4. Try context-based response
    response = no_keys_response(intent, context, prev_intent, prev_topic, confusion)

    if response:
        return jsonify(response)

    return build_response(None, intent, None, confusion)

def handle_confusion(data, intent, topic, confusion, prev_intent, prev_topic):

    if intent == "emotional_support":
        return {
            "response": data["response"],
            "intent": intent,
            "topic": topic,
            "context": "misc",
            "confusion_label": confusion
        }

    if not confusion:
        return build_response(data, intent, topic, confusion)

    # If current topic has confused response
    
    # Check if data has response
    if data.get("response"):
        return build_response(data, intent, topic, confusion, confused=True)

    # fallback to previous topic
    if prev_intent and prev_topic and prev_topic != "clarify":
        prev_data = chatlola_data[prev_intent][prev_topic]
        return jsonify({
            "response": "Mukhang medyo nakakalito 😅. " + prev_data["confused_response"],
            "intent": prev_intent,
            "topic": prev_topic + "_confused",
            "confusion_label": confusion
        })

    return jsonify({
        "response": "Maaari mo bang linawin kung anong bahagi ang gusto mong maintindihan?",
        "intent": None,
        "topic": "clarify",
        "context": "general",
        "confusion_label": confusion
    })

def no_keys_response(intent, context, prev_intent, prev_topic, confusion):
    ctx = context_data.get(context, {})

    if intent == "emotional_support":
        return {
           "response": None,
            "intent": intent,
            "topic": "",
            "context": "misc"
        }

    for topic, data in ctx.items():
        if data["intent"] == intent:
            kb = chatlola_data[intent][topic]

            use_confused = (
                confusion or
                (prev_intent == intent and prev_topic == topic)
            )

            return {
                "response": kb["confused_response"] if use_confused else kb["response"],
                "intent": intent,
                "topic": topic,
                "context": kb["context"]
            }

def clarify_response(query):
    for intent, intent_data in chatlola_data.items():
        for topic, data in intent_data.items():
            if match_keywords(query, data["keywords"]):
                return jsonify({
                    "response": data["response"],
                    "intent": intent,
                    "topic": topic,
                    "context": data["context"]
                })

    return {
        "response": None,
        "intent": None,
        "topic": "clarify",
        "context": "general"
    }