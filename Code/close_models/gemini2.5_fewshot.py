from google import genai
from google.genai import types
import json
import os
from PIL import Image
import base64
from io import BytesIO
import datetime
import concurrent.futures
from tqdm import tqdm
import time
import random

client = genai.Client(api_key="your_api_key")

# Local image directory
IMAGE_DIR = 'your_image_path'

# List of special category prefixes
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

# Configuration parameters
NUM_EXAMPLES = 1 # Number of examples, can be 1, 2 or 3 
ALIGNED_EXAMPLES = True     # Whether to use aligned examples

# Whether to use aligned examples

def encode_image(image_path):
    """Convert local image to bytes for Google GenAI"""
    try:
        with open(image_path, 'rb') as image_file:
            return image_file.read()
    except Exception as e:
        print(f"Encoding error: {e}")
        return None

def call_gemini(prompt: str, image_path):
    """Call Gemini for inference using Google GenAI"""
    try:
        # Read image bytes
        image_bytes = encode_image(image_path)
        if image_bytes is None:
            return None

        # Call the model with the image and prompt
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt
            ]
        )
        return response.text

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def get_last_processed_id(output_file):
    """Get the last processed ID from the output file"""
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            last_id = None
            for line in file:
                try:
                    data = json.loads(line)
                    last_id = data.get('id')
                except json.JSONDecodeError:
                    continue
            return last_id
    except FileNotFoundError:
        return None

def get_category_from_image_name(image_name):
    """Extract category from image name"""
    # Check if it's a special category
    for special_prefix in SPECIAL_CATEGORIES:
        if image_name.startswith(special_prefix):
            return special_prefix
    
    # If not a special category, return the main category
    parts = image_name.split('-')
    if len(parts) >= 1:
        return parts[0]
    
    return None

def get_example_data(current_image_name=None):
    """Get data for few-shot examples"""
    examples = []
    try:
        with open("example.jsonl", 'r', encoding='utf-8') as file:
            all_examples = []
            for line in file:
                try:
                    data = json.loads(line)
                    all_examples.append(data)
                except json.JSONDecodeError:
                    continue
            
            # If current image name is not provided, randomly select from all examples
            if current_image_name is None:
                print(f"Current image name not provided, randomly selecting {NUM_EXAMPLES} examples")
                if len(all_examples) >= NUM_EXAMPLES:
                    return random.sample(all_examples, NUM_EXAMPLES)
                else:
                    return all_examples
            
            # Get the category of the current image
            current_category = get_category_from_image_name(current_image_name)
            
            # Determine if the current image belongs to a special category
            is_current_special = any(current_image_name.startswith(sp) for sp in SPECIAL_CATEGORIES)
            
            # Categorize all examples
            category_examples = {}
            special_examples = []
            
            for example in all_examples:
                example_image = example['image']
                example_is_special = any(example_image.startswith(sp) for sp in SPECIAL_CATEGORIES)
                
                # Collect special category examples
                if example_is_special:
                    special_examples.append(example)
                
                # Collect examples by category
                example_category = get_category_from_image_name(example_image)
                if example_category not in category_examples:
                    category_examples[example_category] = []
                category_examples[example_category].append(example)
            
            # Select the target example set based on alignment settings
            if ALIGNED_EXAMPLES:
                if is_current_special:
                    # If it's a special category, prioritize selecting from the same special category
                    target_examples = [ex for ex in special_examples if ex['image'].startswith(current_category)]
                    # If there are not enough examples of the same special category, supplement from all special examples
                    if len(target_examples) < NUM_EXAMPLES:
                        other_special = [ex for ex in special_examples if not ex['image'].startswith(current_category)]
                        target_examples.extend(random.sample(other_special, min(NUM_EXAMPLES - len(target_examples), len(other_special))))
                else:
                    # If it's a normal category, select from the same category
                    target_examples = category_examples.get(current_category, [])
                    # If there are not enough examples of the same category, supplement from other normal categories
                    if len(target_examples) < NUM_EXAMPLES:
                        other_examples = []
                        for cat, exs in category_examples.items():
                            if cat != current_category and not any(ex['image'].startswith(sp) for ex in exs for sp in SPECIAL_CATEGORIES):
                                other_examples.extend(exs)
                        if other_examples and len(target_examples) < NUM_EXAMPLES:
                            target_examples.extend(random.sample(other_examples, min(NUM_EXAMPLES - len(target_examples), len(other_examples))))
            else:
                # Unaligned case: if the current image is a special category, select from non-special categories; otherwise, select from special categories
                target_examples = [ex for ex in all_examples if any(ex['image'].startswith(sp) for sp in SPECIAL_CATEGORIES) != is_current_special]
            
            # Select the required number of examples from the target examples
            if len(target_examples) <= NUM_EXAMPLES:
                examples = target_examples
            else:
                examples = random.sample(target_examples, NUM_EXAMPLES)
                
    except Exception as e:
        print(f"Error reading example data: {e}")
    
    return examples

def process_single_data(data, image_dir):
    """Process single data item"""
    try:
        data_id = data.get('id', 'unknown')
        image_name = data['image']
        
        print(f"\nProcessing image: {image_name}")
        print(f"Question: {data['question']}")
        
        # Get few-shot example data
        examples = get_example_data(image_name)
        
        # Build few-shot prompt
        prompt = f'You are currently a senior expert in scale recognition.\n' \
                f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option.\n' \
                f'Note that you only need to choose one option from all options without explaining any reason.\n'
        
        if NUM_EXAMPLES > 0:
            prompt += f'Given the following {NUM_EXAMPLES} examples to learn the scale recognition task:\n'
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt += f'Example{i}: Input: Image:<image{i}>\n' \
                    f'Question: {example["question"]}, Options: {"; ".join(example["options"])}.\n' \
                    f'Output: {example["answer"]}\n\n'
        
        # Add current question
        prompt += f'Input: Image:<image>\n' \
                f'Question: {data["question"]}, Options: {"; ".join(data["options"])}.\n' \
                f'Output:'
        
        # Build image path list
        image_paths = []
        # Add example image paths
        for example in examples:
            example_image_path = os.path.join(image_dir, example['image'])
            image_paths.append(example_image_path)
        
        # Add current image path
        current_image_path = os.path.join(image_dir, data['image'])
        image_paths.append(current_image_path)
        
        # Try to call the model
        try:
            # Get example image name list
            example_images = [ex['image'] for ex in examples]
            example_images_str = ", ".join(example_images)
            print(f"Using {len(examples)} examples [{example_images_str}]")
            
            model_answer = call_gemini(prompt, image_paths)
            
            if model_answer is not None:
                # Extract the option (one of ABCD) from the model answer
                lines = model_answer.strip().split('\n')
                output = lines[-1].strip()
                print(f"Model answer: {output}")
                
                if len(output) == 0:
                    output = '--'
                
                # Determine if the answer is correct
                if output.upper() in data['answer'] or data['answer'].upper() in output:
                    return {
                        "id": data_id,
                        "image": data['image'],
                        "options": data['options'],
                        "result": 1,
                        "output": output,
                        "answer": data['answer']
                    }
                else:
                    return {
                        "id": data_id,
                        "image": data['image'],
                        "options": data['options'],
                        "result": 0,
                        "output": output,
                        "answer": data['answer']
                    }
                
        except Exception as e:
            error_msg = f"Model call failed: {e}"
            print(f"ID {data_id} - {error_msg}")
            
            return {
                "id": data_id,
                "image": data['image'],
                "result": 0,
                "output": "ERROR",
                "answer": data['answer'],
                "error": str(e)
            }

    except Exception as e:
        print(f"ID {data_id} - Unexpected error: {e}")
        return {
            "id": data_id,
            "image": data['image'],
            "result": 0,
            "output": "ERROR",
            "answer": data['answer'],
            "error": str(e)
        }

def process_jsonl(input_file, output_file):
    """Process JSONL file and evaluate results"""
    # Get the last processed ID
    last_processed_id = get_last_processed_id(output_file)
    print(f"Last processed ID: {last_processed_id}")
    
    # Read all data
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError:
                continue
    
    # Find the position where processing last stopped
    start_index = 0
    if last_processed_id is not None:
        for i, data in enumerate(all_data):
            if data.get('id') == last_processed_id:
                start_index = i + 1
                break
        print(f"Skipping processed data, starting from ID {last_processed_id}")
    
    # Get data to process
    data_to_process = all_data[start_index:]
    total_count = len(data_to_process)
    print(f"Number of data items to process: {total_count}")
    
    # Create an ordered dictionary to store results
    from collections import OrderedDict
    results = OrderedDict()
    next_id_to_write = start_index
    
    # Track consecutive failures
    consecutive_failures = 0
    
    # Process data using a thread pool to reduce concurrency
    with open(output_file, 'a', encoding='utf-8') as out_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_data = {}
            
            # Use tqdm to create a progress bar
            with tqdm(total=total_count, desc="Processing progress") as pbar:
                # Submit one task at a time, and submit the next one after it's done
                for data in data_to_process:
                    future = executor.submit(process_single_data, data, IMAGE_DIR)
                    future_to_data[future] = data
                    
                    # Wait for the current task to complete
                    try:
                        result = future.result()
                        # Store the result in the ordered dictionary
                        data_id = result['id']
                        results[data_id] = result
                        
                        # Check if it's an error result (including invalid answers and API call failures)
                        if result.get('output') == 'ERROR':
                            consecutive_failures += 1
                            print(f"\nConsecutive failures: {consecutive_failures}")
                            if consecutive_failures >= 3:
                                print("\nThree consecutive failures (invalid answers or API call failures), exiting program")
                                # Ensure all processed results are written
                                while next_id_to_write in results:
                                    result_to_write = results[next_id_to_write]
                                    out_file.write(json.dumps(result_to_write, ensure_ascii=False) + '\n')
                                    out_file.flush()
                                    del results[next_id_to_write]
                                    next_id_to_write += 1
                                return
                        else:
                            consecutive_failures = 0  # Reset consecutive failure count
                        
                        # Write results in order
                        while next_id_to_write in results:
                            result_to_write = results[next_id_to_write]
                            try:
                                out_file.write(json.dumps(result_to_write, ensure_ascii=False) + '\n')
                                out_file.flush()
                                del results[next_id_to_write]
                                next_id_to_write += 1
                                # Only update progress bar after successful data write
                                pbar.update(1)
                            except Exception as e:
                                print(f"\nError writing data: {e}")
                                break
                                
                    except Exception as e:
                        print(f"\nError processing data: {e}")

if __name__ == "__main__":
    # You can modify configuration parameters here
    # NUM_EXAMPLES = 2  # Number of examples, can be 1, 2 or 3
    # ALIGNED_EXAMPLES = True  # Whether to use aligned examples
    
    # Generate output filename based on configuration
    aligned_str = "aligned" if ALIGNED_EXAMPLES else "nonaligned"
    output_filename = f"gemini2.5_fewshot_{NUM_EXAMPLES}examples_{aligned_str}.jsonl"
    
    input_file_path = "test_500.jsonl"
    output_file_path = os.path.join("your_output_path", output_filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Add start processing log
    print(f"{'='*50}")
    print(f"Starting data set processing: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Number of examples: {NUM_EXAMPLES}")
    print(f"Examples aligned: {ALIGNED_EXAMPLES}")
    print(f"Special categories: {SPECIAL_CATEGORIES}")
    print(f"Time: {datetime.datetime.now()}")
    print(f"{'='*50}")
    
    try:
        process_jsonl(input_file_path, output_file_path)
        
        # Add processing completion log
        print(f"{'='*50}")
        print(f"Data processing complete")
        print(f"Time: {datetime.datetime.now()}")
        print(f"{'='*50}")
    except Exception as e:
        # Add error handling
        print(f"{'='*50}")
        print(f"An error occurred during processing: {e}")
        print(f"Time: {datetime.datetime.now()}")
        print(f"{'='*50}")
        
