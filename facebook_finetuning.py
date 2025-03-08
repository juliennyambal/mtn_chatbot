from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load OPT-125M
model_name = "facebook/opt-125m"
FINETUNED_MODEL = "./fine-tuned-opt-125m"
DATASET = 'mtn_chatbot_dataset.json'
TRAINING_LOGS = "./logs"
TRAINING_RESULTS = "./results"
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Loading the dataset
dataset = load_dataset('json', data_files = DATASET)

# First data point
print("Dataset structure:", dataset)
print("First example:", dataset['train'][0] if 'train' in dataset else next(iter(dataset.values()))[0])
print("Column names:", dataset['train'].column_names if 'train' in dataset else next(iter(dataset.values())).column_names)

def tokenize_function(examples):
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
        print("Unexpected structure:", list(examples.keys()))
        first_text_field = next((k for k in examples.keys() 
                               if isinstance(examples[k], list) and examples[k] 
                               and isinstance(examples[k][0], str)), None)
        texts = examples[first_text_field] if first_text_field else ["Placeholder text"]
        
    result = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    
    result["labels"] = result["input_ids"].copy()
    
    return result

# Batched processing
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset['train'].column_names if 'train' in dataset else next(iter(dataset.values())).column_names
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # We're doing causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir=TRAINING_RESULTS,
    per_device_train_batch_size=10,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_steps=500,
    save_total_limit=2,
    logging_dir=TRAINING_LOGS,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'] if 'train' in tokenized_datasets else next(iter(tokenized_datasets.values())),
    data_collator=data_collator,  
)

trainer.train()

model.save_pretrained(FINETUNED_MODEL)
tokenizer.save_pretrained(FINETUNED_MODEL)