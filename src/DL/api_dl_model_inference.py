import requests
from pprint import pprint
API_URL = 'https://api-inference.huggingface.co/models/bigscience/bloomz'
headers = {'Authorization': 'ENTER_THE_ACCESS_KEY_HERE'}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

params = {'max_length': 200, 'top_k': 10, 'temperature': 2.5}
output = query({'inputs': 'Sherlock Holmes is a', 'parameters': params, })
print(output)

[{'generated_text': 'Sherlock Holmes is a private investigator whose cases ' 'have inspired several film productions'}]