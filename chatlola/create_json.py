import json

chatlola_data = {}

intents = [
    'what_is',
    'how_to_prevent_scam',
    'scam_response',
    'scam_scenarios',
    'emotional_support'
] #replacing this later

for intent in intents:
    chatlola_data["intents"][intent] = {
        "tags": []
    }

with open("chatlola/chatlola_data.json", "w") as f:
    json.dump(chatlola_data, f, indent=4)