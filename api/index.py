#run locally with: python -m api.index
from flask import Flask, request, jsonify
from chatlola.dialog_manager import intent_recognition, conversation_management
import string
import json

app = Flask(__name__)

@app.route('/respond', methods=['GET'])

def respond():
    query = request.args.get('query')

    query = query.translate(str.maketrans('', '', string.punctuation))
    query = query.lower()
    
    intent = intent_recognition(query)
    
    response_data = conversation_management(query, intent)

    return jsonify({k: v for k, v in response_data.items() if k != "keywords"})

#get specific response for when user clicks a suggestion    
@app.route('/getresponse', methods=['GET'])

def getresponse():
    intent = request.args.get('intent')
    tag = request.args.get('tag')

    with open("chatlola/knowledge_base.json", "r") as file:
        chatlola_data = json.load(file)
    
    response_data = chatlola_data[intent][tag]
    
    return jsonify({k: v for k, v in response_data.items() if k != "keywords"})

if __name__=='__main__':
    app.run(debug=True)
