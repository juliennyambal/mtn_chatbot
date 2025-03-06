from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

# Load your dataset
dataset = load_dataset('csv', data_files='mtn_chatbot_dataset.csv')

# Create a mapping from action names to numeric labels
action_to_label = {}
unique_actions = set(dataset['train']['Action'])
for i, action in enumerate(unique_actions):
    action_to_label[action] = i

# Save the mapping for later use
import json
with open('action_to_label.json', 'w') as f:
    json.dump(action_to_label, f)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_function(examples):
    # Convert text actions to numeric labels
    labels = [action_to_label[action] for action in examples['Action']]
    
    # Tokenize with padding and truncation to a fixed length
    result = tokenizer(examples['User Query'], padding="max_length", truncation=True, max_length=64)
    
    # Add labels to the result
    result['labels'] = labels
    
    return result

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for training
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(min(1000, len(tokenized_datasets['train']))))
eval_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(min(200, len(tokenized_datasets['train']))))

# Load the TinyBERT model with the correct number of labels
num_labels = len(action_to_label)
model = AutoModelForSequenceClassification.from_pretrained(
    'huawei-noah/TinyBERT_General_4L_312D', 
    num_labels=num_labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Save artifacts
model.save_pretrained('./fine-tuned-tinybert')
tokenizer.save_pretrained('./fine-tuned-tinybert')