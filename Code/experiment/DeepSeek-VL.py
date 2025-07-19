import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import json
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from PIL import Image

DEVICE_INDEX = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = "your_input_file_path"  # Input file path
RESULT_FILE_PATH = "your_output_file_path"  # Output file path
IMAGE_DIR = "your_image_dir"  # Image directory

# Load the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(DEVICE_INDEX).to(torch.bfloat16).cuda().eval()

# Initialize counters
count = 0
right_count = 0

# Process input file and generate results
with open(f'{FILE_PATH}test.jsonl', 'r', encoding="utf-8") as f, open(f'{RESULT_FILE_PATH}deepseek.jsonl', 'w+', encoding="utf-8") as fout:
    for line in f:
        data = json.loads(line)
        question = data['question']
        id = data['id']
        options = data['options']
        image_name = data['image']
        image_filepath = IMAGE_DIR + image_name
        
        # Load the image
        pil_image = Image.open(image_filepath)
        
        # Build the conversation format
        conversation = [
            {
                "role": "User",
                "content": f'Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason.Image: <image_placeholder>, Question: {question}, Options: {"; ".join(options)}. \nOutput:',
                "images": [image_filepath]
            },
            {
                "role": "Assistant",
                "content": "You are currently a senior expert in scale recognition. ",
            }
        ]

        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # Run the image encoder to get image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        # Decode the output
        generated_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip().rstrip('.')
        print(generated_text)
        output = generated_text
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
