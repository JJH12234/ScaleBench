import os
import json
import torch
from PIL import Image
from modelscope import AutoConfig, AutoModel, AutoTokenizer
# from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'mPLUG/mPLUG-Owl3-7B-240728'
IMAGE_DIR = "your_image_dir"  
JSON_PATH = "test.jsonl"
OUTPUT_PATH = "mplug.jsonl"

# Load the model
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = model.to(device).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = model.init_processor(tokenizer)
# config = mPLUGOwl3Config.from_pretrained(MODEL_PATH)
# print(config)
# # model = mPLUGOwl3Model(config).cuda().half()
# model = mPLUGOwl3Model.from_pretrained(MODEL_PATH, attn_implementation='sdpa', torch_dtype=torch.half)
# model.to(device).eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# processor = model.init_processor(tokenizer)

count = 0
right_count = 0
# Process the input file and generate results
with open(JSON_PATH, 'r', encoding="utf-8") as f, open(OUTPUT_PATH, 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        print(f"Processing item {id}, ID: {question}")
        image = Image.open(os.path.join(IMAGE_DIR, image_name)).convert('RGB')
    
        prompt=f"<|image|>You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options}. \nOutput:"
        # Inference
        messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]

        inputs = processor(messages, images=[image], videos=None)

        inputs.to('cuda')
        inputs.update({
            'tokenizer': tokenizer,
            'max_new_tokens':100,
            'decode_text':True,
        })

        output = model.generate(**inputs)
        # If output is a list, take the first element and convert it to a string
        if isinstance(output, list):
            output = str(output[0]) if output else "--"
        else:
            output = str(output)
            
        print(output)
        count += 1
        
        # Process the result and write to the output file
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
    
    
