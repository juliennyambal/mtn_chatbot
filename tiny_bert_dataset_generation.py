import pandas as pd
import random

# Define possible actions and their corresponding queries
actions = {
    "Send money": ["Send {amount} ZAR to {recipient}", "Transfer {amount} ZAR to {recipient}", "I want to send {amount} ZAR to {recipient}"],
    "Check balance": ["Check my balance", "What is my current balance?", "How much money do I have?"],
    "Pay bill": ["Pay my {bill} bill", "I need to pay {amount} ZAR for my {bill}", "Pay {amount} ZAR for {bill}"],
    "Apply for loan": ["Apply for a loan of {amount} ZAR", "I want to borrow {amount} ZAR", "Can I get a loan for {amount} ZAR?"],
    "Check loan": ["How much do I owe on my loan?", "What is my loan balance?", "Check my loan status"],
    "Transfer money": ["Transfer {amount} ZAR to my {account}", "Move {amount} ZAR to {account}", "Send {amount} ZAR to {account}"],
}

# Define possible recipients, bills, and accounts
recipients = ["John", "Sarah", "Mike", "Jane", "David", "Emma"]
bills = ["electricity", "water", "internet", "rent", "phone"]
accounts = ["savings", "checking", "investment"]

# Generate 2000 data points
data = []
for _ in range(2000):
    action = random.choice(list(actions.keys()))
    if action == "Send money":
        amount = random.randint(50, 10000)
        recipient = random.choice(recipients)
        query = random.choice(actions[action]).format(amount=amount, recipient=recipient)
        data.append([query, action, amount, recipient])
    elif action == "Check balance":
        query = random.choice(actions[action])
        data.append([query, action, None, None])
    elif action == "Pay bill":
        amount = random.randint(100, 5000)
        bill = random.choice(bills)
        query = random.choice(actions[action]).format(amount=amount, bill=bill)
        data.append([query, action, amount, bill])
    elif action == "Apply for loan":
        amount = random.randint(1000, 50000)
        query = random.choice(actions[action]).format(amount=amount)
        data.append([query, action, amount, None])
    elif action == "Check loan":
        query = random.choice(actions[action])
        data.append([query, action, None, None])
    elif action == "Transfer money":
        amount = random.randint(50, 10000)
        account = random.choice(accounts)
        query = random.choice(actions[action]).format(amount=amount, account=account)
        data.append([query, action, amount, account])

# Create a DataFrame
df = pd.DataFrame(data, columns=["User Query", "Action", "Amount", "Recipient"])

# Save to CSV
df.to_csv("mtn_chatbot_dataset.csv", index=False)