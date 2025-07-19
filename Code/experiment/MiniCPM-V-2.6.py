import os
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Path configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'openbmb/MiniCPM-V-2_6'
IMAGE_DIR = "your_image_dir"  
JSON_PATH = "test.jsonl"
OUTPUT_PATH = "minicpm.jsonl"


# Load model and tokenizer
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=torch.float16
)
# attn_implementation='eager'
model = model.to(device).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# TARGET_SIZE = (448, 448)  # Or (224, 224), adjust according to the model's requirements


def main():
    count = 0
    right_count = 0
    # Process input file and generate results
    with open(JSON_PATH, 'r', encoding="utf-8") as f, open(OUTPUT_PATH, 'w+', encoding="utf-8") as fout:
        for line in f:
            data = json.loads(line)
            question = data['question']
            id = data['id']
            options = data['options']
            image_name = data['image']
            image_filepath = IMAGE_DIR + image_name
            
            # print(f"Processing item {id}, ID: {question}")
            images = Image.open(os.path.join(IMAGE_DIR, image_name)).convert('RGB')
            prompt=f"You are currently a senior expert in scale recognition. Given an Image, a Question and Options, your task is to identify the scale value and select the correct option. Note that you only need to choose one option from all options without explaining any reason. Input: Question: {question}{options}. \nOutput:"
            # Inference
            msgs = [{'role': 'user', 'content': [images,prompt]}]

            # Inference
            try:
                res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
                output = res if isinstance(res, str) else str(res)
            except Exception as e:
                print(f"Inference error: {e}, image: {images}")
                output = ""
            
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

if __name__ == "__main__":
    main()