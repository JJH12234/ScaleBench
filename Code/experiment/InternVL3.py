from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import json
import torch
from PIL import Image

DEVICE_INDEX = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory


model_checkpoint = "OpenGVLab/InternVL3-8B-hf"
#model_checkpoint = "OpenGVLab/InternVL3-14B-hf"
#model_checkpoint = "OpenGVLab/InternVL3-38B-hf"
#model_checkpoint = "OpenGVLab/InternVL3-78B-hf"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=DEVICE_INDEX, torch_dtype=torch.bfloat16)

# Initialize counters
count = 0
right_count = 0

# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}internvl3.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        # # Load image
        # pil_image = Image.open(image_filepath)
        
        # Construct conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_filepath},
                    {"type": "text", "text": f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}, Options: {"; ".join(options)}. \nPlease answer directly to option letter.'},
                    
                ],
            },
        ]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        generate_ids = model.generate(**inputs, max_new_tokens=500)
        output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        
        
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

