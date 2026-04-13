#run locally with: python -m api.index
from flask import Flask, request, jsonify
from chatlola.dialog_manager import intent_recognition, confusion_detection, conversation_management, clarify_response
import string
import json

app = Flask(__name__)

@app.route('/respond', methods=['GET'])

def respond():
    query = request.args.get('query')

    query = query.translate(str.maketrans('', '', string.punctuation))
    query = query.lower()

    #previous topic
    prev_intent = request.args.get('intent')
    prev_topic = request.args.get('topic')

    context = request.args.get('context')

    #if in clarify mode
    if prev_topic == "clarify":
        return clarify_response(query)
    
    intent = intent_recognition(query)
    confusion_label = confusion_detection(query)

    response_data = conversation_management(query, intent, confusion_label, prev_intent, prev_topic, context)

    return response_data

if __name__=='__main__':
    app.run(debug=True)
