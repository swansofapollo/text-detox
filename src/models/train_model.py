from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict

# Load your data from the preprocessed CSV file
data = load_dataset('csv', data_files='../data/preprocessed_data.csv')['train']

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')

def tokenize_function(examples):
    return tokenizer(examples['reference'], examples['translation'], 
                     max_length=128, truncation=True, padding='max_length')

def prepare_data(examples):
    # Tokenize the reference texts
    model_inputs = tokenizer(examples["reference"], max_length=128, truncation=True, padding="max_length")

    # Tokenize the translation texts with the same tokenizer but do not pad yet, as we need raw token ids for labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["translation"], max_length=128, truncation=True)["input_ids"]

    # Pad labels to max_length
    labels = [label + [tokenizer.pad_token_id] * (128 - len(label)) for label in labels]

    model_inputs["labels"] = labels

    return model_inputs

# Tokenize and prepare the data
tokenized_data = data.map(tokenize_function, batched=True)
model_data = tokenized_data.map(prepare_data, batched=True)

# Remove unnecessary columns
columns_to_remove = ['reference', 'translation']
for column in columns_to_remove:
    if column in model_data.features:
        model_data = model_data.remove_columns(column)

# Split the dataset
dataset = DatasetDict({
    'train': model_data.train_test_split(test_size=0.1)['train'],
    'validation': model_data.train_test_split(test_size=0.1)['test']
})

# Initialize the T5 model for sequence-to-sequence LM
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Data collator used for dynamically padding the inputs and labels
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('models/model1')
