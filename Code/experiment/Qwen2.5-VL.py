import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH= 'Qwen/Qwen2.5-VL-7B-Instruct'
#MODEL_PATH= 'Qwen/Qwen2.5-VL-32B-Instruct'
#MODEL_PATH= 'Qwen/Qwen2.5-VL-72B-Instruct'
FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto",
).half()
#  device_map="auto"
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
#processor = AutoProcessor.from_pretrained("/home/ecust/Models/Qwen2.5-VL-7B")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1400*28*28
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)
count = 0
right_count = 0
# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}qwen25vl.jsonl', 'w+', encoding="utf-8") as fout:
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
                    {"type": "text", "text": f'You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}, Options: {"; ".join(options)}. '}
                ]
            }
        ]

        # Prepare inference input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generate output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
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
