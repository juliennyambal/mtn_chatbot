from transformers import pipeline
import json

# Load the action to label mapping
with open('action_to_label.json', 'r') as f:
    action_to_label = json.load(f)

# Create the reverse mapping (label to action)
label_to_action = {v: k for k, v in action_to_label.items()}

def map_label_to_action(result):
    # Extract the label index from LABEL_X format
    label_index = int(result[0]['label'].split('_')[1])
    
    # Map to the corresponding action
    if label_index in list(label_to_action.keys()):
        action = label_to_action[label_index]
    else:
        action = f"Unknown action (index: {label_index})"
    
    return {
        "action": action,
        "confidence": result[0]['score']
    }  

class Infere:

    # Function to map label to action
    @staticmethod
    def predict(query):
        # Load the fine-tuned model
        classifier = pipeline('text-classification', model='./fine-tuned-tinybert')
        result = classifier(query)
        mapped_result = map_label_to_action(result)
        return result ,mapped_result

    
# Test the model with mapping
test_queries = [
    "Pay 323 ZAR for water",
    "Can I get a loan for 6371 ZAR?",
    "I want to send 61 ZAR to David"
]


if __name__ == "__main__":
    for query in test_queries:
        result, mapped_result = Infere.predict(query)
        
        print(f"Query: {query}")
        print(f"Raw result: {result}")
        print(f"Mapped action: {mapped_result['action']}")
        print(f"Confidence: {mapped_result['confidence']:.4f}")
        print("-" * 50)

#