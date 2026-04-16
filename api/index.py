#run locally with: python -m api.index
from flask import Flask, request, jsonify
from chatlola.dialog_manager import intent_recognition, confusion_detection, conversation_management, clarify_response
from chatlola.tts import existing_tts, getTTS, uploadAudio
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

@app.route('/tts', methods=['GET'])

def tts():
    text = request.args.get('text')
    public_id = request.args.get('public_id')

    url = existing_tts(public_id)

    if url:
        return { "url": url }

    audio = getTTS(text)
    
    if audio: 
        url = uploadAudio(audio, public_id)

    if url:
        return { "url": url }
    
    return { "url": None }

if __name__=='__main__':
    app.run(debug=True)
