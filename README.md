<br />
<p align="center">
  <h1 align="center"> 🔭Tiny Scales, Great Challenges: The Limits of Multimodal LLMs in Scale
Recognition </h1>
  <h3 align="center">ScaleBench: A new benchmark dataset for scale recognition of MLLMs.</h3>
  
  <p align="center">  
<!--     <a href="">arxiv</a> -->
    ·
    <a href="https://github.com/JJH12234/ScaleBench/blob/main/Dateset/dataset/train.jsonl">github</a>
    ·
    <a href="https://github.com/JJH12234/ScaleBench/blob/main/LICENSE">license</a>
<!--     <a href="">benchmark</a> -->
    
  </p>
</p>


## Contents

- [ScaleBench](#Contents)
  - [Overview](#1-Overview)
    - [Examples](#Examples)
    - [Detail Information](#Detail-Information)
  - [Access ScaleBench](#2-Access-ScaleBench)
    - [Data Split](#Data-Split)
    - [Data Format](#Data-Format)
  - [Experiment & Evaluation](#3-Experiment-and-Evaluation)
    - [Experiment](#Experiment)
    - [Evaluation](#Evaluation)
  - [License](#4-License)




## 1 Overview
**ScaleBench** is a **manually annotated** dataset designed for **multimodal visual scale recognition** in a **m**ultiple-choice **q**uestion.
The dataset comprises 6,574 samples and 5,371 images, with 1,112 images associated with multiple questions.Our benchmark covers common units and measurement objects across 13 major categories of physical quantities, aiming to ensure a diverse set of questions. We report the frequency of different units and the
number of measurement objects for each physical quantity.To address the limitations of existing datasets, we clearly define annotation guidelines for ScaleBench.
### Examples
The following figures list some classic examples(500 images) in our dataset. You can click out [`Examples`](Examples) to view partial details of the dataset.

### Detail Information
The following table [`Splits/`](Comparison/splits.png) lists the detailed information statistics of the splited dataset.
<br>
You can find our dataset through the following path **_(Dataset/dataset)_** for more details.
<br>
_Due to the fact that only redirecting to the specified file is valid in anonymous links, redirecting to the specified directory is invalid. Therefore, we use bold and italicized font to indicate the markings of all specified directories, making it easier for reviewers to search. Thank you!_


## 2 Access ScaleBench
Due to anonymity requirements, our dataset is temporarily stored in [`Images`](Images). It will be published on [huggingface](https://huggingface.co) later.

###  Data Split
As reported in the folloeing table, ScaleBench contains 6,574 samples, divided into training, validation, and test sets according to a 7:1:2 ratio.
<br>All the splited data sets are in the directory **_(Dataset/dataset)_**. 


### Data Format
Each `jsonl` file is of the following format:
```json
{"image": "time-clock-00146.jpg", "question": "What is the time shown by the clock in the image?", "options": ["A.9:27:32", "B.9:32:27", "C.9:31:27", "D.9:27:31"], "answer": "C"}
{"image": "angles-compass-00011.jpg", "question": "What is the degree that the white pointer of the compass in the image is pointing to?", "options": ["A.280", "B.290", "C.300", "D.310"], "answer": "B"}
{"image": "speed-speedometer-00022.jpg", "question": "What is the speed shown by the speedometer in the image in km/h?", "options": ["A.105", "B.100", "C.115", "D.110"], "answer": "C"}
{"..."}
```
Each line is an individual data point.
`image`  denotes name of the image . `question`  is the question with manual annotation, `options`  is reasonable numerical options.
<br>
ScaleBench covers 13 commonly used physical quantities, spanning 33 types of measurement objects and 38 scale units, thereby aligning more closely with real-world application scenarios.You can see all of them in the file [`U & O type/`](Dataset/type/Units_and_Object.png). 

## 3 Experiment and Evaluation
### Experiment
We have disclosed the inference code for the model in the directory **_(Code/experiment)_**,  as well as the fine-tuning code in the directory **_(Code/finetune)_**.
<br>
- For all 9 open-sourse MLLMs, you can directly execute Python files in the directory **_(Code/experiment)_** to perform inference on models before and after fine-tuning: 
```
nohup python DeepSeek-VL.py > log/DeepSeek-VL.log 2>1& &
nohup python InternVL3.py > log/InternVL3.log 2>1& &
nohup python Janus-Pro.py > log/Janus-Pro.log 2>1& &
nohup python Llama-3.2-Vision.py > log/Llama-3.2-Vision.log 2>1& &
nohup python LLaVA-v1.6.py > log/LLaVA-v1.6.log 2>1& &
nohup python MiniCPM-V-2.6.py > log/MiniCPM-V-2.6.log 2>1& &
nohup python mPLUG-Owl3.py > log/mPLUG-Owl3.log 2>1& &
nohup python Phi-3.5-vision.py > log/Phi-3.5-vision.log 2>1& &
nohup python Qwen2.5-VL.py > log/Qwen2.5-VL.log 2>1& &
```
Due to the large amount of open source model code, you need to download it yourself through channels or call it directly from platforms such as [huggingface](https://huggingface.co).
- For open-source models, You can execute Bash files using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [ms-swift](https://github.com/modelscope/ms-swift) in the directory **_(Code/finetune)_** to perform fine-tuning:
```
nohup bash DeepSeek-VL.sh > log/DeepSeek-VL_train.log 2>1& &
nohup bash InternVL3.sh > log/InternVL3_train.log 2>1& &
nohup bash Janus-Pro.sh > log/Janus-Pro_train.log 2>1& &
nohup bash Llama-3.2-Vision.sh > log/Llama-3.2-Vision_train.log 2>1& &
nohup bash LLaVA-v1.6.sh > log/LLaVA-v1.6_train.log 2>1& &
nohup bash MiniCPM-V-2.6.sh > log/MiniCPM-V-2.6_train.log 2>1& &
nohup bash mPLUG-Owl3.sh > log/mPLUG-Owl3_train.log 2>1& &
nohup bash Phi-3.5-vision.sh > log/Phi-3.5-vision_train.log 2>1& &
nohup bash Qwen2.5-VL.sh > log/Qwen2.5-VL_train.log 2>1& &
```
- For gemini-2.5-pro and gpt-4o, you can directly execute our Python file in the directory **_(Code/close_models)_** to perform inferencing of the zero-shot, few-shot, provided that you prepare a key:
```
python gemini2.5_zeroshot.py
python gemini2.5_fewshot.py
python gpt4o_zeroshot.py
python gpt4o_fewshot.py
```
Gemini-2.5-pro needs to apply on the [official website](https://aistudio.google.com/app/apikey), and GPT-4o needs to be purchased on the [official website](https://openai.com/).

### Evaluation
You can process the results of model inference through the code we provide to calculate overall accuracy,the accuracy of each physical quantity category, overall P, R, F1 indicators,. We integrate the calculation process into the Python files in the directory **_(Code/eval)_**:
```
python calculate_prf1.py
python calculate_acc.py
```

### Requirements
The environment configuration required for debugging code is placed in directory **_(Code/requirement)_**

## 4 License
This project is licensed under the [Apache-2.0 License](LICENSE).
