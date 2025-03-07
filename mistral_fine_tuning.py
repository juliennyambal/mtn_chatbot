from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datasets import load_dataset

# Load Mistral 7B for CPU (no quantization)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Starting model download...")
# Load model specifically for CPU - avoid quantization which is GPU-focused
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cpu",
    torch_dtype=torch.float32,  # Use full precision for CPU
    low_cpu_mem_usage=True      # Helps with memory management
)
print("Model downloaded successfully")

# Load your dataset
print("Loading dataset...")
dataset = load_dataset('json', data_files='mtn_chatbot_dataset.json')
print("Dataset loaded")

# Ensure tokenizer has proper padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Adjust based on your actual dataset structure
    # Print an example to see structure
    print("Dataset example:", examples[0] if len(examples) > 0 else "Empty")
    
    # Adapt this based on your dataset structure
    inputs = [f"{ex['instruction']}\n{ex['input']}" if 'input' in ex else ex['instruction'] 
              for ex in examples]
    
    return tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512
    )

# First check dataset structure before mapping
print("Dataset structure check:", dataset.keys())
print("First example:", dataset['train'][0] if 'train' in dataset else next(iter(dataset.values()))[0])

# Map tokenization
tokenized_datasets = dataset['train'].map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset['train'].column_names  # Remove all original columns
)
print("Dataset tokenized")

# Define training arguments for CPU
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Keep small for CPU
    gradient_accumulation_steps=8,  # Increase for CPU
    num_train_epochs=1,            # Reduce for testing
    learning_rate=2e-5,
    fp16=False,                    # Disable mixed precision on CPU
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    no_cuda=True                   # Explicitly disable CUDA
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Fine-tune the model
print("Starting training...")
trainer.train()
print("Training complete")

model.save_pretrained("./fine-tuned-mistral-7b")
tokenizer.save_pretrained("./fine-tuned-mistral-7b")