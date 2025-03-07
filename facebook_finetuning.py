from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load OPT-125M
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Make sure the tokenizer has padding token set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load your dataset
dataset = load_dataset('json', data_files='mtn_chatbot_dataset.json')

# Print dataset info to understand structure
print("Dataset structure:", dataset)
print("First example:", dataset['train'][0] if 'train' in dataset else next(iter(dataset.values()))[0])
print("Column names:", dataset['train'].column_names if 'train' in dataset else next(iter(dataset.values())).column_names)

# Adjust the tokenization function based on your dataset structure
def tokenize_function(examples):
    # Replace these field names with your actual dataset column names
    if 'text' in examples:
        texts = examples['text']
    elif 'instruction' in examples:
        if 'input' in examples and 'output' in examples:
            texts = [f"{inst}\n{inp}\n{out}" for inst, inp, out in 
                   zip(examples['instruction'], examples['input'], examples['output'])]
        elif 'output' in examples:
            texts = [f"{inst}\n{out}" for inst, out in 
                   zip(examples['instruction'], examples['output'])]
        else:
            texts = examples['instruction']
    else:
        # If different structure, modify this part
        print("Unexpected structure:", list(examples.keys()))
        # Fallback to first text field
        first_text_field = next((k for k in examples.keys() 
                               if isinstance(examples[k], list) and examples[k] 
                               and isinstance(examples[k][0], str)), None)
        texts = examples[first_text_field] if first_text_field else ["Placeholder text"]
        
    # Important: For causal LM training we need labels to be the same as input_ids
    result = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    
    # This is critical for the model to compute loss properly
    result["labels"] = result["input_ids"].copy()
    
    return result

# Use batched processing
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset['train'].column_names if 'train' in dataset else next(iter(dataset.values())).column_names
)

# Use a data collator that handles language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # We're doing causal LM, not masked LM
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=10,
    num_train_epochs=5,
    learning_rate=1e-5,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Define the Trainer with the data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'] if 'train' in tokenized_datasets else next(iter(tokenized_datasets.values())),
    data_collator=data_collator,  # This ensures proper loss computation
)

# Fine-tune the model
trainer.train()

model.save_pretrained("./fine-tuned-opt-125m")
tokenizer.save_pretrained("./fine-tuned-opt-125m")