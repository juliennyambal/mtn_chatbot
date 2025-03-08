from transformers import pipeline, AutoTokenizer

# Load the tokenizer from the fine-tuned model directory
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-opt-125m")

# Load the fine-tuned model
classifier = pipeline(
    "text-generation",
    model="./fine-tuned-opt-125m",
    tokenizer=tokenizer,
    device=-1  # Use CPU
)

class Infere:

    # Function to map label to action
    @staticmethod
    def predict(prompt):
        query = "I need to send money to my friend, Marc"
        prompt = f"Classify the intent and extract parameters from the user query.\n{query}"
        # Generate response with adjusted parameters
        response = classifier(
            prompt,
            max_new_tokens=50,  # Adjust based on expected output length
            # temperature=0.7,    # Reduce randomness
            do_sample=False,    # Use greedy decoding
            top_k=50,           # Limit the number of high-probability tokens
            # top_p=0.9           # Use nucleus sampling
        )
        return response

if __name__ == "__main__":

    # Test the model
    query = "I need to send money to my friend, Marc"
    prompt = f"Classify the intent and extract parameters from the user query.\n{query}"

    # Generate response with adjusted parameters
    response = classifier(
        prompt,
        max_new_tokens=50,  # Adjust based on expected output length
        # temperature=0.7,    # Reduce randomness
        do_sample=False,    # Use greedy decoding
        top_k=50,           # Limit the number of high-probability tokens
        # top_p=0.9           # Use nucleus sampling
    )

    print(response)