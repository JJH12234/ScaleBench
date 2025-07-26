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

def process_single_data(data, image_dir):
    """Process single data item"""
    try:
        data_id = data.get('id', 'unknown')
        print(f"\nProcessing image: {data['image']}")
        print(f"Question: {data['question']}")
        
        
        prompt = f'You are currently a senior expert in scale recognition.\n' \
                f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option.\n' \
                f'Note that you only need to choose one option from all options without explaining any reason.\n' \
                f'Input: Image:<image>, Question: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput:' 
        # Construct image path
        image_path = os.path.join(image_dir, data['image'])
        
        # Try calling the model (max 3 attempts)
        for attempt in range(3):
            try:
                model_answer = call_gemini(prompt, image_path)
                if model_answer is not None:
                    
                    lines = model_answer.strip().split('\n')
                    output = lines[-1].strip()
                    print(output)
                    
                    if len(output) == 0:
                        output = '--'
                    if output.upper() in data['answer'] or data['answer'].upper() in output:
                        return {
                            "id": data_id,
                            "image": data['image'],
                            "result": 1,
                            "output": output,
                            "answer": data['answer']
                        }
                    else:
                        return {
                            "id": data_id,
                            "image": data['image'],
                            "result": 0,
                            "output": output,
                            "answer": data['answer']
                        }
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {e}"
                print(f"ID {data_id} - {error_msg}")
                
                if attempt == 2:  # Last attempt failed
                    return {
                        "id": data_id,
                        "image": data['image'],
                        "result": 0,
                        "output": "ERROR",
                        "answer": data['answer'],
                        "error": "ERROR"
                    }

    except Exception as e:
        print(f"ID {data_id} - Unexpected error: {e}")
        return {
            "id": data_id,
            "image": data['image'],
            "result": 0,
            "output": "ERROR",
            "answer": data['answer'],
            "error": "ERROR"
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
    
    # Find the position of the last processed data
    start_index = 0
    if last_processed_id is not None:
        for i, data in enumerate(all_data):
            if data.get('id') == last_processed_id:
                start_index = i + 1
                break
        print(f"Skipping processed data, starting from ID {last_processed_id} and continuing")
    
    # Get data to be processed
    data_to_process = all_data[start_index:]
    total_count = len(data_to_process)
    print(f"Number of items to process: {total_count}")
    
    # Create an ordered dict to store results
    from collections import OrderedDict
    results = OrderedDict()
    next_id_to_write = start_index + 1 if start_index < len(all_data) else 0
    
    # Track consecutive failures
    consecutive_failures = 0
    
    # Use thread pool to process data, reduce concurrency
    with open(output_file, 'a', encoding='utf-8') as out_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_data = {}
            
            # Use tqdm to create progress bar
            with tqdm(total=total_count, desc="Processing progress") as pbar:
                # Submit one task at a time, wait for it to complete, then submit the next
                for data in data_to_process:
                    future = executor.submit(process_single_data, data, IMAGE_DIR)
                    future_to_data[future] = data
                    
                    # Wait for the current task to complete
                    try:
                        result = future.result()
                        # Store the result in the ordered dict
                        data_id = result['id']
                        results[data_id] = result
                        
                        # Check if it's an error result (including invalid answer and API call failure)
                        if result.get('output') == 'ERROR':
                            consecutive_failures += 1
                            print(f"\nConsecutive failures: {consecutive_failures}")
                            if consecutive_failures >= 3:
                                print("\nThree consecutive failures (invalid answer or API call failure), exiting program")
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
    input_file_path = "test_500.jsonl"
    output_file_path = "gemini_0_shot.jsonl"
    process_jsonl(input_file_path, output_file_path)