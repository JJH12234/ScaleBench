from openai import OpenAI
import json
import os
from PIL import Image
import base64
from io import BytesIO
import datetime
import concurrent.futures
from tqdm import tqdm
import random
import time

client = OpenAI(
    api_key='your_api_key',
    base_url='your_base_url'
)

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
NUM_EXAMPLES = 1  # Number of examples, can be 1, 2 or 3
ALIGNED_EXAMPLES = False  # Whether to use aligned examples

def encode_image(image_path):
    """Convert local image to base64 encoding"""
    try:
        with open(image_path, 'rb') as image_file:
            img_str = base64.b64encode(image_file.read()).decode('utf-8')
            return 'data:image/jpeg;base64,' + img_str
    except Exception as e:
        print(f"Encoding error: {e}")
        return None

def call_gpt4(prompt: str, image_path, detail='auto'):
    """Call GPT-4 for inference"""
    try:
        # Encode image
        image_encoded = encode_image(image_path)
        if image_encoded is None:
            return None

        # Add random delay to avoid frequent requests
        time.sleep(random.uniform(2.0, 5.0))
        
        # Maximum 3 retries
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}] \
                                    + [{"type": "image_url", "image_url": { "url": image_encoded, "detail": detail}}]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.5,
                )
                
                # Handle possible string response or object response
                if isinstance(response, str):
                    return response.strip()
                else:
                    # Normal object response handling
                    return response.choices[0].message.content.strip()
                    
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {retry+1}/{max_retries} failed: {error_message}")
                
                # If not the last attempt, wait before retrying
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached, giving up on this request")
                    return None

    except Exception as e:
        print(f"Error during answering: {e}")
        return None

def get_last_processed_id(output_file):
    """Get the last processed ID from output file"""
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
    
    # If not special category, return main category
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
            
            # If no current image name provided, randomly select from all examples
            if current_image_name is None:
                print(f"No image name provided, randomly selecting {NUM_EXAMPLES} examples from all")
                if len(all_examples) >= NUM_EXAMPLES:
                    return random.sample(all_examples, NUM_EXAMPLES)
                else:
                    return all_examples
            
            # Get current image's category
            current_category = get_category_from_image_name(current_image_name)
            
            # Check if current image belongs to special category
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
            
            # Select target examples based on alignment setting
            if ALIGNED_EXAMPLES:
                if is_current_special:
                    # If special category, prioritize examples from same special category
                    target_examples = [ex for ex in special_examples if ex['image'].startswith(current_category)]
                    # If not enough same-category examples, supplement from other special examples
                    if len(target_examples) < NUM_EXAMPLES:
                        other_special = [ex for ex in special_examples if not ex['image'].startswith(current_category)]
                        target_examples.extend(random.sample(other_special, min(NUM_EXAMPLES - len(target_examples), len(other_special))))
                else:
                    # If normal category, select from same category
                    target_examples = category_examples.get(current_category, [])
                    # If not enough same-category examples, supplement from other normal categories
                    if len(target_examples) < NUM_EXAMPLES:
                        other_examples = []
                        for cat, exs in category_examples.items():
                            if cat != current_category and not any(ex['image'].startswith(sp) for ex in exs for sp in SPECIAL_CATEGORIES):
                                other_examples.extend(exs)
                        if other_examples and len(target_examples) < NUM_EXAMPLES:
                            target_examples.extend(random.sample(other_examples, min(NUM_EXAMPLES - len(target_examples), len(other_examples))))
            else:
                # Non-aligned case: if current is special, select from non-special; otherwise select from special
                target_examples = [ex for ex in all_examples if any(ex['image'].startswith(sp) for sp in SPECIAL_CATEGORIES) != is_current_special]
            
            # Select required number of examples from target examples
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
        
        # Get few-shot example data
        examples = get_example_data(image_name)
        
        # Build few-shot prompt
        prompt = f'You are currently a senior expert in scale recognition.\n' \
                f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option.\n'
                f'Note that you only need to choose one option from all options without explaining any reason.\n'
        
        if NUM_EXAMPLES > 0:
            prompt += f'Given the following {NUM_EXAMPLES} examples to learn the scale recognition task\n'
        
        # Add examples
        for i, example in enumerate(examples, 1):
            example_image_path = os.path.join(image_dir, example['image'])
            example_image_encoded = encode_image(example_image_path)
            # Add example information (image will be added in the message array later)
            prompt += f'Example{i}: Input: Image:<image>\n' \
                    f'Question: {example["question"]}, Options: {"; ".join(example["options"])}.\n' \
                    f'Output: {example["answer"]}\n'
        
        # Add current question
        prompt += f'Input: Image:<image> Question: {data["question"]}, Options: {"; ".join(data["options"])}.\n' \
                f'Output:'
        
        # Build image path
        image_path = os.path.join(image_dir, data['image'])
        
        # Directly call model (internal retry mechanism)
        try:
            # Prepare message content, including text and image
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Add example images
            for example in examples:
                example_image_path = os.path.join(image_dir, example['image'])
                example_image_encoded = encode_image(example_image_path)
                if example_image_encoded:
                    messages[0]["content"].append({"type": "image_url", "image_url": {"url": example_image_encoded, "detail": "auto"}})
            
            # Add current question image
            current_image_encoded = encode_image(image_path)
            if current_image_encoded:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": current_image_encoded, "detail": "auto"}})
            
            # Add random delay to avoid frequent requests
            time.sleep(random.uniform(2.0, 5.0))
            
            # Maximum 3 retries
            max_retries = 3
            model_answer = None
            
            for retry in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.5,
                    )
                    
                    # Handle possible string response or object response
                    if isinstance(response, str):
                        model_answer = response.strip()
                    else:
                        # Normal object response handling
                        model_answer = response.choices[0].message.content.strip()
                        #print(f"Processing data ID: {data_id}, Image: {image_name}, used {len(examples)} examples, model answer: {model_answer}")
                        # Get list of example image names
                        example_images = [ex['image'] for ex in examples]
                        example_images_str = ", ".join(example_images)
                        
                        print(f"Processing data ID: {data_id}, Image: {image_name}, used {len(examples)} examples [{example_images_str}], model answer: {model_answer}")
                    
                    break  # Successfully got answer, exit loop
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Processing data ID: {data_id}, Attempt {retry+1}/{max_retries} failed: {error_message}")
                    
                    # If not the last attempt, wait before retrying
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 2  # Exponential backoff
                        print(f"Processing data ID: {data_id}, waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                    else:
                        print(f"Processing data ID: {data_id}, Max retries reached, giving up on this request")
            
            if model_answer is None:
                error_msg = "Model call failed, empty result returned"
                print(f"Processing data ID: {data_id} - {error_msg}")
                
                return {
                    "id": data_id,
                    "image": data['image'],
                    "result": 0,
                    "output": "--",
                    "answer": data['answer'],
                    "error": "Failed to get model answer"
                }
                
        except Exception as e:
            error_msg = f"Model call exception: {e}"
            print(f"Processing data ID: {data_id} - {error_msg}")
            
            
            return {
                "id": data_id,
                "image": data['image'],
                "result": 0,
                "output": "--",
                "answer": data['answer'],
                "error": str(e)
            }

        # Process model output
        try:
            output = model_answer.strip().rstrip('.')
            # Check if answer is correct
            if output in data['answer'].upper():
                return {
                    "id": data_id,
                    "image": data['image'],
                    "options": data['options'],
                    "result": 1,
                    "output": output,
                    "answer": data['answer']
                }
            elif data['answer'] in output.upper():
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
            print(f"Processing data ID: {data_id} - Error processing output: {e}")
            return None
            
    except Exception as e:
        print(f"Processing data ID: {data_id} - Unexpected error: {e}")
        return None

def process_jsonl(input_file, output_file):
    """Process JSONL file and evaluate results"""
    # Get last processed ID
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
    
    # Find the position of the last processed data
    start_index = 0
    if last_processed_id is not None:
        for i, data in enumerate(all_data):
            if data.get('id') == last_processed_id:
                start_index = i + 1
                break
        print(f"Skipping processed data, starting from ID {last_processed_id} onwards")
    
    # Get data to process
    data_to_process = all_data[start_index:]
    print(f"Number of data items to process: {len(data_to_process)}")
    
    # Create an ordered dictionary to store results
    from collections import OrderedDict
    results = OrderedDict()
    next_id_to_write = start_index  # Modify here, starting from start_index, not start_index + 1
    
    # Process data using a thread pool
    with open(output_file, 'a', encoding='utf-8') as out_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_data = {
                executor.submit(process_single_data, data, IMAGE_DIR): data 
                for data in data_to_process
            }
            
            # Use tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(future_to_data), 
                             total=len(data_to_process),
                             desc="Processing progress"):
                data = future_to_data[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Store result in ordered dictionary
                        data_id = result['id']
                        results[data_id] = result
                        
                        # Write results in order
                        while next_id_to_write in results:
                            result_to_write = results[next_id_to_write]
                            out_file.write(json.dumps(result_to_write, ensure_ascii=False) + '\n')
                            out_file.flush()
                            del results[next_id_to_write]
                            next_id_to_write += 1
                            
                except Exception as e:
                    print(f"Error processing data: {e}")
                    continue

if __name__ == "__main__":
    # You can modify configuration parameters here
    # NUM_EXAMPLES = 2  # Number of examples, can be 1, 2 or 3
    # ALIGNED_EXAMPLES = True  # Whether to use aligned examples
    
    # Generate output filename based on configuration
    aligned_str = "aligned" if ALIGNED_EXAMPLES else "nonaligned"
    output_filename = f"gpt4o_fewshot_{NUM_EXAMPLES}examples_{aligned_str}.jsonl"
    
    input_file_path = "test_500.jsonl"
    output_file_path = os.path.join("your_output_path", output_filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Add start processing log
    print(f"{'='*50}")
    print(f"Starting data processing: {input_file_path}")
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
        print(f"Data processing completed")
        print(f"Time: {datetime.datetime.now()}")
        print(f"{'='*50}")
    except Exception as e:
        # Add error handling
        print(f"{'='*50}")
        print(f"Error during processing: {e}")
        print(f"Time: {datetime.datetime.now()}")
        print(f"{'='*50}")
        