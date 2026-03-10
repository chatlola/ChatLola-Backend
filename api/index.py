#run locally with: python -m api.index
from flask import Flask, request, jsonify
from chatlola.chatlola import intent_recognition, conversation_management
import string
import json

app = Flask(__name__)

@app.route('/respond', methods=['GET'])

def respond():
    query = request.args.get('query')

    query = query.translate(str.maketrans('', '', string.punctuation))
    query = query.lower()
    
    intent = intent_recognition(query)
    
    response, related_responses = conversation_management(query, intent)

    return jsonify({"response": response, "related_responses": related_responses})

#get specific response for when user clicks a suggestion    
@app.route('/getresponse', methods=['GET'])

def getresponse():
    intent = request.args.get('intent')
    tag = request.args.get('tag')

    with open("chatlola/chatlola_data.json", "r") as file:
        chatlola_data = json.load(file)
    
    response_data = chatlola_data[intent][tag]
    
    return jsonify({"response": response_data["response"], "related_responses": response_data["related"]})

if __name__=='__main__':
    app.run(debug=True)
