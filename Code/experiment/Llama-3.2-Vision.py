import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

count = 0
right_count = 0
# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}llama3.2.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        # Build conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_filepath},
                    {"type": "text", "text": f'You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}, Options: {"; ".join(options)}. \nPlease answer directly to option letter. '}
                ]
            }
        ]

        # Open local image
        image = Image.open(image_filepath)

        # Prepare inference input
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        # Generate output
        generated_id = model.generate(**inputs, max_new_tokens=30)
        full_output = processor.decode(generated_id[0]).strip()
        
        # Extract the actual answer, keeping only the option letter
        try:
            # First, extract the assistant's response part from the full output
            assistant_part = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            raw_answer = assistant_part.split("<|eot_id|>")[0].strip()
            
            # Further extract the option letter (A/B/C/D)
            option_match = re.search(r'(?:^|\s|\n)([A-D])(?:[\.。]|\s|$)', raw_answer, re.IGNORECASE)
            if option_match:
                output = option_match.group(1).upper()  # Extract only the option letter and convert to uppercase
            else:
                # If no clear option is found, try to match option words
                text_option_match = re.search(r'(?:answer is|correct answer is|select|option)\s*([A-D])', raw_answer, re.IGNORECASE)
                if text_option_match:
                    output = text_option_match.group(1).upper()
                else:
                    output = raw_answer  # If extraction fails, keep the original answer
        except:
            # If extraction fails, keep the original output
            output = full_output
        
        print(output)
        count += 1
        
        # Process results and write to output file
        if len(output) == 0:
            output = '--'
        if output.upper() in data['answer']:
            result_json = {'id': id, "image":data['image'], 'result': 1, "output": output, "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            right_count += 1
        elif data['answer'] in output.upper():
            result_json = {'id': id, "image":data['image'], 'result': 1, "output": output, "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')
            right_count += 1
        else:
            result_json = {'id': id, "image":data['image'], 'result': 0, "output": output, "answer": data['answer']}
            fout.write(json.dumps(result_json, ensure_ascii=False) + '\n')

# Print accuracy
if count > 0:
    accuracy = right_count / count
    print(f"Accuracy: {accuracy:.2f} ({right_count}/{count})")
    

