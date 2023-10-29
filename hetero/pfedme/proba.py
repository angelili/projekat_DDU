import json
import math

# Read the JSON file
with open('training_history_acc_dist_pfedme.json', 'r') as json_file:
    data = json.load(json_file)

# Access and inspect 'accuracy_local_pfedme'
accuracy_local_pfedme = data['accuracy_local_pfedme']

# Print the values of 'accuracy_local_pfedme'
print(len(accuracy_local_pfedme))
