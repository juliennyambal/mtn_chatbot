from datasets import load_dataset, Dataset
from tqdm import tqdm
import ollama

# # Load the local CSV file
# dataset = load_dataset('csv', data_files='momo_conversations.csv', split='train')

# # Inspect the dataset
# print(dataset[35])  # Print the first row

MTN_MOMO_SYSTEM_PROMPT = """You are the MTN Mobile Money virtual assistant for Uganda. Be helpful, professional, and knowledgeable about all MTN Mobile Money services. Use clear, simple language and provide accurate information about savings, loans, transfers, and account services.
Always verify user identity for sensitive operations, offer relevant financial advice, and suggest appropriate MTN products based on the customer's needs and usage patterns. Maintain a friendly but professional tone, and prioritize educating customers about financial services while ensuring security compliance."""

CLEANING_PROMPT = """
Your task is to standardize MTN Mobile Money conversational data.
Remove any system actions, technical notations, or contextual descriptions while preserving the actual dialogue. Here are some examples:

Input: verifies customer identity. Your MTN Mobile Money balance is UGX 245,600. You can check your transaction history by dialing *165*5#.
Output: Your MTN Mobile Money balance is UGX 245,600. You can check your transaction history by dialing *165*5#.

Input: processes payment request. I've sent UGX 50,000 to phone number 0751234567. The transaction fee was UGX 1,200.
Output: I've sent UGX 50,000 to phone number 0751234567. The transaction fee was UGX 1,200.

Input: checks loan eligibility. Based on your transaction history, you qualify for a loan of up to UGX 300,000 with a 30-day repayment period.
Output: Based on your transaction history, you qualify for a loan of up to UGX 300,000 with a 30-day repayment period.
"""

def load_mtn_mobile_money_dataset(dataset_location='dataset/mtn-mobile-money-conversations-*'):
    """Loads the MTN Mobile Money conversation dataset.

    This function loads the MTN Mobile Money conversation dataset from the specified
    source, which contains customer-bot interactions related to mobile money services in Uganda.

    Returns:
        datasets.Dataset: A dataset containing MTN Mobile Money conversations.
            The dataset includes columns for dialogue, speaker (Customer/Bot),
            scenario type (Savings, Loans, Remittance, Money Transfers, Account Status),
            and conversation ID.
    """
    # Load the local CSV file
    dataset = load_dataset('csv', data_files=dataset_location, split='train')
    return dataset

def create_conversation_pairs(dataset):
    """Creates conversation pairs from the MTN Mobile Money conversation dataset.

    This function processes the dataset to create conversation pairs where a Customer
    message is followed by the Bot's response. Each conversation includes a system prompt
    defining the Bot's personality and purpose.

    Args:
        dataset (datasets.Dataset): The MTN Mobile Money dataset containing dialogue,
            speaker, scenario type, and conversation ID information.

    Returns:
        datasets.Dataset: A new dataset containing conversation pairs in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": mtn_momo_system_prompt},
                    {"from": "human", "value": customer_dialogue},
                    {"from": "gpt", "value": bot_dialogue}
                ]
            }
    """
    new_rows = []
    # Group by conversation_id to handle multi-turn conversations
    conversation_groups = {}

    for i in tqdm(range(len(dataset))):
        row = dataset[i]
        conv_id = row["conversation_id"]

        if conv_id not in conversation_groups:
            conversation_groups[conv_id] = []
        conversation_groups[conv_id].append(row)

    # Process each conversation
    for conv_id, conversation in conversation_groups.items():
        for i in range(len(conversation) - 1):
            current_row = conversation[i]
            next_row = conversation[i + 1]

            if current_row["speaker"] == "Customer" and next_row["speaker"] == "Bot":
                new_rows.append(
                    {
                        "conversations_raw": [
                            {"from": "system", "value": MTN_MOMO_SYSTEM_PROMPT.strip()},
                            {"from": "human", "value": current_row["dialogue"].strip()},
                            {"from": "gpt", "value": next_row["dialogue"].strip()},
                        ],
                        "scenario_type": current_row["scenario_type"]
                    }
                )

    return Dataset.from_list(new_rows)

def clean_dialogue(text, system_prompt):
    """Clean a single dialogue using Ollama.

    Args:
        text (str): The dialogue text to clean.
        system_prompt (str): The system prompt providing instructions for cleaning.

    Returns:
        str: The cleaned dialogue text with actions/context removed.
    """
    response = ollama.chat(
        model="mistral",  # Change this to "llama3" or another available model in Ollama
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text.strip()},
        ]  # Lower temperature for more deterministic responses
    )

    return response["message"]["content"]  # Extracts the chatbot's reply


def clean_conversations(dataset):
    """Clean all conversations in the dataset by removing action descriptions and context.

    This function processes each conversation in the dataset by removing action descriptions,
    stage directions, and other contextual information from the dialogue, leaving only the
    spoken lines.

    Args:
        dataset (datasets.Dataset): The input dataset containing conversations in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": str},
                    {"from": "human", "value": str},
                    {"from": "gpt", "value": str}
                ]
            }

    Returns:
        datasets.Dataset: A new dataset with cleaned conversations in the format:
            {
                "conversations": [
                    {"from": "system", "value": str},
                    {"from": "human", "value": str},
                    {"from": "gpt", "value": str}
                ]
            }
    """
    new_rows = []

    for row in tqdm(dataset):
        rick_completion = clean_dialogue(row["conversations_raw"][1]["value"], CLEANING_PROMPT
        )
        non_rick_completion = clean_dialogue(row["conversations_raw"][2]["value"], CLEANING_PROMPT
        )

        new_rows.append(
            {
                "conversations": [
                    {"from": "system", "value": row["conversations_raw"][0]["value"]},
                    {"from": "human", "value": rick_completion},
                    {"from": "gpt", "value": non_rick_completion},
                ]
            }
        )

    return Dataset.from_list(new_rows)

def main():
    print("Loading dataset...")
    dataset_created = load_mtn_mobile_money_dataset()
    print("Number of rows: ", len(dataset_created))
    print("Creating conversation pairs...")
    sharegpt_dataset = create_conversation_pairs(dataset_created)
    print("Cleaning conversations...")
    cleaned_dataset = clean_conversations(sharegpt_dataset)
    cleaned_dataset.save_to_disk("./mtn_bot_dataset/sharegpt_momo_dataset")

if __name__ == "__main__":
    main()