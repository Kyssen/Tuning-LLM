def index_to_letter(index):
    return chr(65 + int(index))  # 0 -> 'A', 1 -> 'B', etc.

def fix_evaluation_results():
    # Read the original file
    with open('evaluation_results.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process the file
    new_lines = []
    total_correct = 0
    total_questions = 0
    current_split = None
    split_correct = 0
    split_total = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for split header
        if line.strip().endswith("Split Results"):
            current_split = line.strip().split()[0]
            split_correct = 0
            split_total = 0
            new_lines.append(line)
            
        # Check for correct answer line
        elif line.strip().startswith("Correct Answer:"):
            numeric_answer = line.strip().split()[-1]
            try:
                letter_answer = index_to_letter(numeric_answer)
                new_lines.append(f"Correct Answer: {letter_answer}\n")
                
                # Get the model answer from next line
                model_answer_line = lines[i + 1]
                model_answer = model_answer_line.strip().split()[-1]
                
                # Check if correct and update counts
                is_correct = model_answer == letter_answer
                split_total += 1
                total_questions += 1
                if is_correct:
                    split_correct += 1
                    total_correct += 1
                
                # Update the "Is Correct" line
                new_lines.append(model_answer_line)
                new_lines.append(f"Is Correct: {is_correct}\n")
                i += 2  # Skip the next two lines we just processed
                
            except (ValueError, IndexError):
                new_lines.append(line)
                
        # Update accuracy statistics
        elif line.strip().startswith("Total Questions:"):
            if current_split:
                new_lines.append(f"Total Questions: {split_total}\n")
                new_lines.append(f"Correct Answers: {split_correct}\n")
                accuracy = (split_correct / split_total * 100) if split_total > 0 else 0
                new_lines.append(f"Accuracy: {accuracy:.2f}%\n")
                i += 2  # Skip the next two lines (old correct answers and accuracy)
            else:
                new_lines.append(line)
        
        # Keep other lines as is
        else:
            new_lines.append(line)
        
        i += 1

    # Write the corrected file
    with open('evaluation_results_fixed.txt', 'w', encoding='utf-8') as f:
        # Write overall summary first
        f.write("Complete Evaluation Results\n")
        f.write("=========================\n\n")
        f.write("Overall Summary:\n")
        f.write("--------------\n")
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        f.write(f"Total Questions Across All Splits: {total_questions}\n")
        f.write(f"Total Correct Answers: {total_correct}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n\n")
        
        # Write the rest of the file
        f.writelines(new_lines)

if __name__ == "__main__":
    fix_evaluation_results()
    print("Results have been fixed and saved to 'evaluation_results_fixed.txt'") 