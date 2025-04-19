import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

def main():
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("kyssen/Unlearning")
    print(f"Dataset loaded with {len(dataset['train'])} examples")

    # Initialize tokenizer and model
    print("Initializing model...")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check for existing checkpoint
    checkpoint_dir = "./story_model"
    if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
        print("Found existing checkpoint. Loading...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    else:
        print("No checkpoint found. Starting from base model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["story"],
            padding=True,
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True
        )

    # Process dataset
    print("Processing dataset...")
    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Calculate total steps for proper warmup
    num_epochs = 1
    batch_size = 1
    total_steps = (len(tokenized_dataset) * num_epochs) // batch_size

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=False,  # Don't overwrite existing checkpoints
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,           # Save more frequently
        save_total_limit=3,      # Keep last 3 checkpoints
        prediction_loss_only=True,
        remove_unused_columns=True,
        logging_first_step=True,
        no_cuda=False,
        warmup_steps=total_steps//10,
        lr_scheduler_type="cosine_with_restarts",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        seed=42,
        bf16=True,
        # Resume training settings
        # resume_from_checkpoint=True,  # Enable checkpoint resumption
        save_safetensors=True,       # Save in safetensors format for faster loading
        # load_best_model_at_end=True  # Load best model at end of training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # Train
    print("Starting/Resuming training...")
    trainer.train(resume_from_checkpoint=False)  # Enable checkpoint resumption

    # Save
    print("Saving model...")
    trainer.save_model(checkpoint_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()