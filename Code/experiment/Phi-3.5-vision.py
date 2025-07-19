import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import re
from PIL import Image 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory
model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='eager'    
)

# For best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=16  # Single-frame images use 16
) 
count = 0
right_count = 0
# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}phi3.5.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        count += 1
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        # Load image
        image = Image.open(image_filepath)
        
        # Build placeholder
        placeholder = "<|image_1|>\n"
        
        # Build conversation format
        messages = [
            {"role": "user", "content": placeholder + f'You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}, Options: {"; ".join(options)}. \nPlease answer directly to option letter.'}
        ]

        # Prepare inference input
        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Process image and text input
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 1000, 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # Remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        output = processor.batch_decode(generate_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False)[0] 

        print(output)
            
        # Process result and write to output file
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
    

