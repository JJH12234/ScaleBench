from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import json
DEVICE_INDEX = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory
model_path = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to(DEVICE_INDEX)

# Initialize counters
count = 0
right_count = 0

# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}llava_v1.6.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        # Load the image
        pil_image = Image.open(image_filepath)
        
        # Build conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}, Options: {"; ".join(options)}. \nPlease answer directly to option letter.'},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(DEVICE_INDEX)

        # Autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=1000)

        output = processor.decode(output[0], skip_special_tokens=True)
        
        # Process output to keep only the answer after [/INST]
        if '[/INST]' in output:
            output = output.split('[/INST]')[1].strip()
        
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

