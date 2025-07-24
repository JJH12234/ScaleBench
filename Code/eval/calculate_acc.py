import json
import os
from collections import defaultdict

# Define constants
INPUT_FOLDER = "your_input_folder"
OUTPUT_FOLDER = "your_output_folder"
OUTPUT_SUFFIX = ""

# Define the list of special category prefixes
SPECIAL_CATEGORIES = [
    "amps-ammeter",
    "angles-protractor",
    "ph-phmeter",
    "pressure-sphygmometer",
    "sound-intensitymeter",
    "speed-anemometer",
    "volts-voltmeter",
    "volume-testtube",
    "weight-steelyard",
    "weight-bodyweigh"
]

def processData(input_file, output_file):
    # Read JSONL file
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error occurred at line {line_num}:")
                    print(f"Error line content: {line}")
                    print(f"Error details: {str(e)}")
                    raise
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        raise
    except Exception as e:
        print(f"Error occurred while reading file: {str(e)}")
        raise
    
    # Initialize counters
    main_category_count = defaultdict(int)  # Number of questions in main category
    sub_category_count = defaultdict(int)   # Number of questions in sub category
    main_category_correct = defaultdict(int)  # Number of correct answers in main category
    sub_category_correct = defaultdict(int)   # Number of correct answers in sub category
    
    # Counters for special categories
    special_count = 0
    special_correct = 0
    
    # Process each data item as a group
    for item in data:
        # Get image name
        image_name = item['image']
        
        # Check if it belongs to special category
        is_special = False
        for special_prefix in SPECIAL_CATEGORIES:
            if image_name.startswith(special_prefix):
                is_special = True
                special_count += 1
                if item['result'] == 1:
                    special_correct += 1
                break
        
        # If not a special category, count into main and sub categories
        if not is_special:
            main_category = image_name.split('-')[0]
            sub_category = image_name.split('-')[1]
            
            # Update counters
            main_category_count[main_category] += 1
            sub_category_count[sub_category] += 1
            
            # Check if the result is correct
            if item['result'] == 1:
                main_category_correct[main_category] += 1
                sub_category_correct[sub_category] += 1
    
    # Calculate overall accuracy
    total_questions = len(data)
    total_correct = sum(1 for item in data if item['result'] == 1)
    total_acc = total_correct / total_questions if total_questions > 0 else 0
    
    # Calculate accuracy for special categories
    special_acc = special_correct / special_count if special_count > 0 else 0
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {total_acc:.4f} ({total_correct}/{total_questions})\n\n")
        
        # Output accuracy for special categories
        f.write(f"Special Category Accuracy: {special_acc:.4f} ({special_correct}/{special_count})\n\n")
        
        # Output by main category
        for main_category in sorted(main_category_count.keys()):
            f.write(f"Main Category\n{main_category}: {main_category_correct[main_category]/main_category_count[main_category]:.4f} ({main_category_correct[main_category]}/{main_category_count[main_category]})\n")
            
            # Find corresponding sub categories
            for sub_category in sorted(sub_category_count.keys()):
                # Check if this sub category belongs to the current main category
                if any(f"{main_category}-{sub_category}" in item['image'] for item in data if not any(item['image'].startswith(sp) for sp in SPECIAL_CATEGORIES)):
                    f.write(f"Sub Category:\n{sub_category}: {sub_category_correct[sub_category]/sub_category_count[sub_category]:.4f} ({sub_category_correct[sub_category]}/{sub_category_count[sub_category]})\n\n")

def processFolder(input_folder, output_folder, output_suffix):
    """
    Process all jsonl files in the folder
    
    Args:
    input_folder: Input folder path
    output_folder: Output folder path
    output_suffix: Output file name suffix
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get all jsonl files
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"Error: No jsonl files found in {input_folder}")
        return
    
    # Process each file
    for jsonl_file in jsonl_files:
        input_path = os.path.join(input_folder, jsonl_file)
        
        # Extract full file name from file name (remove .jsonl suffix)
        model_name = os.path.splitext(jsonl_file)[0]
        output_file_name = f"{model_name}{output_suffix}.txt"
        output_path = os.path.join(output_folder, output_file_name)
        
        print(f"Processing file: {jsonl_file}")
        print(f"Outputting results to: {output_file_name}")
        
        try:
            processData(input_path, output_path)
            print(f"Successfully processed {jsonl_file}")
        except Exception as e:
            print(f"Error occurred while processing {jsonl_file}: {str(e)}")

if __name__ == "__main__":
    processFolder(INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_SUFFIX)
    print(f"All files processed, results are saved in the {OUTPUT_FOLDER} folder")
