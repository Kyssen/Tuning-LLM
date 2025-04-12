import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

def index_to_letter(index):
    return chr(65 + int(index))  # 0 -> 'A', 1 -> 'B', etc.

def format_question(question, choices):
    # Only provide the question and choices, not the story
    formatted = f"### Instruction:\nAnswer the following multiple choice question.\n\n### Input:\nQuestion: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(65 + i)}. {choice}\n"
    formatted += "\n### Response:\n"
    return formatted

def evaluate_split(model, tokenizer, dataset, split_name, device, results_file):
    print(f"\nEvaluating {split_name} split (10% sample)...")
    correct = 0
    total = 0
    results = []
    
    # First, collect all questions
    all_questions = []
    for item in dataset:
        for q in item['questions']:
            all_questions.append(q)  # Only store the question, not the story
    
    # Randomly sample 10% of questions
    sample_size = max(1, int(len(all_questions) * 0.1))  # At least 1 question
    sampled_questions = random.sample(all_questions, sample_size)
    
    print(f"Evaluating {sample_size} questions out of {len(all_questions)} total")

    for q in tqdm(sampled_questions):
        question = q['question']
        choices = q['choices']
        correct_answer = index_to_letter(q['correct_answer'])  # Convert to letter immediately

        # Format the question without story context
        prompt = format_question(question, choices)

        # Generate response
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Extract the answer (assuming model responds with A, B, C, or D)
        model_answer = response[0] if response and response[0] in 'ABCD' else 'Invalid'
        
        # Check if correct
        is_correct = (model_answer == correct_answer)
        if is_correct:
            correct += 1
        total += 1

        # Store result (without story)
        results.append({
            'question': question,
            'choices': choices,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'is_correct': is_correct,
            'full_response': response
        })

    # Calculate accuracy
    accuracy = (correct / total) * 100

    # Write results for this split
    results_file.write(f"\n{split_name} Split Results (10% sample)\n")
    results_file.write("="*50 + "\n")
    results_file.write(f"Sample Size: {total} questions (10% of {len(all_questions)} total)\n")
    results_file.write(f"Correct Answers: {correct}\n")
    results_file.write(f"Accuracy: {accuracy:.2f}%\n\n")
    results_file.write(f"Detailed {split_name} Results:\n")
    results_file.write("-"*50 + "\n\n")
    
    for i, result in enumerate(results, 1):
        results_file.write(f"Question {i}:\n")
        results_file.write(f"Question:\n{result['question']}\n\n")
        results_file.write("Choices:\n")
        for j, choice in enumerate(result['choices']):
            results_file.write(f"{chr(65 + j)}. {choice}\n")
        results_file.write(f"\nCorrect Answer: {result['correct_answer']}\n")
        results_file.write(f"Model Answer: {result['model_answer']}\n")
        results_file.write(f"Is Correct: {result['is_correct']}\n")
        results_file.write("\nComplete Model Response:\n")
        results_file.write("-" * 30 + "\n")
        results_file.write(f"{result['full_response']}\n")
        results_file.write("-" * 30 + "\n")
        results_file.write("\n" + "="*50 + "\n\n")

    return accuracy, total, correct, len(all_questions)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the aligned model
    print("Loading aligned model...")
    model_path = "./aligned_model/final"  # Changed to use the aligned model
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set up tokenizer properly
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Ensure proper padding for causal LM
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Load all splits of the dataset
    print("Loading all splits of the dataset...")
    dataset = load_dataset("kyssen/unlearning")
    
    # Open results file
    with open('aligned_model_results_10percent.txt', 'w', encoding='utf-8') as f:
        f.write("Aligned Model Evaluation Results (10% sample)\n")
        f.write("=====================================\n\n")
        
        # Evaluate each split
        split_results = {}
        total_questions_full = 0
        for split_name in dataset.keys():
            accuracy, total, correct, full_size = evaluate_split(model, tokenizer, dataset[split_name], split_name, device, f)
            split_results[split_name] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "full_size": full_size
            }
            total_questions_full += full_size
        
        # Write overall summary at the top
        f.seek(0)
        f.write("Aligned Model Evaluation Results (10% sample)\n")
        f.write("=====================================\n\n")
        f.write("Overall Summary:\n")
        f.write("--------------\n")
        total_sampled = sum(r["total"] for r in split_results.values())
        total_correct = sum(r["correct"] for r in split_results.values())
        overall_accuracy = (total_correct / total_sampled * 100) if total_sampled > 0 else 0
        f.write(f"Total Questions in Dataset: {total_questions_full}\n")
        f.write(f"Questions Evaluated (10% sample): {total_sampled}\n")
        f.write(f"Total Correct Answers: {total_correct}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n\n")
        
        for split_name, results in split_results.items():
            f.write(f"{split_name} Split:\n")
            f.write(f"  Total Questions in Split: {results['full_size']}\n")
            f.write(f"  Questions Evaluated: {results['total']}\n")
            f.write(f"  Correct Answers: {results['correct']}\n")
            f.write(f"  Accuracy: {results['accuracy']:.2f}%\n\n")

    print("\nEvaluation complete! Results have been saved to 'aligned_model_results_10percent.txt'")

if __name__ == "__main__":
    main() 