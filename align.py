import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

def main():
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load the story-trained model
    print("Loading story-trained model...")
    model_path = "./story_model"
    if not os.path.exists(model_path):
        raise ValueError("Story model not found! Please run train.py first.")
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load Alpaca dataset
    print("Loading Alpaca dataset...")
    full_dataset = load_dataset("tatsu-lab/alpaca")
    # Take only 20% of the training data
    dataset = full_dataset["train"].shuffle(seed=42).select(range(int(len(full_dataset["train"]) * 0.2)))
    print(f"Using {len(dataset)} examples (20% of full dataset)")

    # Format and tokenize the dataset
    def format_and_tokenize(example):
        """Format the instruction and tokenize"""
        if example["input"]:
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        
        # Tokenize the text
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None  # Return list instead of tensors
        )
        return tokenized

    print("Formatting and tokenizing dataset...")
    processed_dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # Split dataset into train and eval
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./aligned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=100,           # Log less frequently (was 10)
        save_strategy="steps",
        save_steps=500,             # Save less frequently (was 50)
        save_total_limit=1,         # Keep only the last checkpoint
        optim="adamw_torch",
        fp16=True,
        eval_strategy="steps",      # Enable evaluation
        eval_steps=500,             # Evaluate less frequently (was 50)
        logging_dir="./logs"
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
    )

    try:
        # Train
        print("\nStarting training...")
        trainer.train()

        # Evaluate final model
        print("\nEvaluating final model...")
        metrics = trainer.evaluate()
        print(f"Final validation loss: {metrics['eval_loss']:.4f}")

        # Save the final model
        print("Saving final model...")
        trainer.save_model("./aligned_model/final")
        print("Training complete!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving current model state...")
        trainer.save_model("./aligned_model/interrupted")
        print("Model saved to ./aligned_model/interrupted")
        try:
            # Try to get validation loss before exiting
            metrics = trainer.evaluate()
            print(f"Validation loss at interruption: {metrics['eval_loss']:.4f}")
        except:
            print("Could not evaluate model at interruption.")
        raise

if __name__ == "__main__":
    main() 