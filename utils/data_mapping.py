import json

def create_mapping(data):
    return json.dumps(data, indent=4)

def save_mapping(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
