# import openai
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
# count = 0
# right_count = 0

def encode_image(image_path):
    """Convert local image to base64 encoding"""
    try:
        with open(image_path, 'rb') as image_file:
            img_str = base64.b64encode(image_file.read()).decode('utf-8')
            return 'data:image/jpeg;base64,' + img_str
    except Exception as e:
        print(f"encoding error: {e}")
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
        
        # Try up to 3 times
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
                
                # Handle string response or object response
                if isinstance(response, str):
                    return response.strip()
                else:
                    # Normal object response handling
                    return response.choices[0].message.content.strip()
                    
            except Exception as e:
                error_message = str(e)
                print(f"Attempt {retry+1}/{max_retries} failed: {error_message}")
                
                
                # If not the last attempt, wait and retry
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # Exponential backoff
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print("Reached max retries, abandoning request")
                    return None

    except Exception as e:
        print(f"Error during answering: {e}")
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

def process_single_data(data, image_dir):
    """Process a single data entry"""
    try:
        data_id = data.get('id', 'unknown')
        prompt = f'You are currently a senior expert in scale recognition.\n' \
                f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option.\n'
                f'Note that you only need to choose one option from all options without explaining any reason.\n'
                f'Input: Image:<image>, Question: {data["question"]}, Options: {"; ".join(data["options"])}. \nOutput:' 
        # Construct image path
        image_path = os.path.join(image_dir, data['image'])
        
        # Call model directly (internal retry mechanism)
        try:
            model_answer = call_gpt4(prompt, image_path)
            if model_answer is None:
                error_msg = "Model call failed, empty result returned"
                print(f"ID {data_id} - {error_msg}")
                
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
            print(f"ID {data_id} - {error_msg}")
            
            # Log error message to error.txt
            with open('error.txt', 'a', encoding='utf-8') as error_file:
                error_file.write(f"\n{'='*50}\n")
                error_file.write(f"Time: {datetime.datetime.now()}\n")
                error_file.write(f"Data ID: {data_id}\n")
                error_file.write(f"Image: {data['image']}\n")
                error_file.write(f"Question: {data['question']}\n")
                error_file.write(f"Options: {data['options']}\n")
                error_file.write(f"Correct answer: {data['answer']}\n")
                error_file.write(f"Error message: {error_msg}\n")
                error_file.write(f"{'='*50}\n")
            
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
            # print(f"Processing ID {data_id}, output: {output}")
            # Check if the answer is correct
            if output in data['answer'].upper():
                return {
                    "id": data_id,
                    "image": data['image'],
                    "result": 1,
                    "output": output,
                    "answer": data['answer']
                }
            elif data['answer'] in output.upper():
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
            print(f"ID {data_id} - Error processing output: {e}")
            # Log error when processing output
            with open('error.txt', 'a', encoding='utf-8') as error_file:
                error_file.write(f"\n{'='*50}\n")
                error_file.write(f"Time: {datetime.datetime.now()}\n")
                error_file.write(f"Data ID: {data_id}\n")
                error_file.write(f"Image: {data['image']}\n")
                error_file.write(f"Question: {data['question']}\n")
                error_file.write(f"Options: {data['options']}\n")
                error_file.write(f"Correct answer: {data['answer']}\n")
                error_file.write(f"Error message: Error processing output: {e}\n")
                error_file.write(f"{'='*50}\n")
            return None
            
    except Exception as e:
        print(f"ID {data_id} - Unexpected error: {e}")
        return None

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
        print(f"Skipping processed data, starting from ID {last_processed_id}")
    
    # Get data to process
    data_to_process = all_data[start_index:]
    print(f"Number of data to process: {len(data_to_process)}")
    
    # Create an ordered dictionary to store results
    from collections import OrderedDict
    results = OrderedDict()
    # Modify here to ensure next_id_to_write starts from the correct ID
    next_id_to_write = start_index
    
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
                        # Store result in the ordered dictionary
                        data_id = result['id']
                        results[data_id] = result
                        
                        # Write results in order
                        while str(next_id_to_write) in results or next_id_to_write in results:
                            # Try both possible key formats (string or integer)
                            result_key = str(next_id_to_write) if str(next_id_to_write) in results else next_id_to_write
                            result_to_write = results[result_key]
                            out_file.write(json.dumps(result_to_write, ensure_ascii=False) + '\n')
                            out_file.flush()
                            del results[result_key]
                            next_id_to_write += 1
                            
                except Exception as e:
                    print(f"Error processing data: {e}")
                    continue

if __name__ == "__main__":
    input_file_path = "test_500.jsonl"
    output_file_path = "gpt4o_0_shot.jsonl"
    process_jsonl(input_file_path, output_file_path)
        
