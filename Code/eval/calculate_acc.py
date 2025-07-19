import json
import os
from collections import defaultdict

# 定义常量
INPUT_FOLDER = r"E:\桌面\project\ScaleDataset_eval\Model_result\biggermodel\result"
OUTPUT_FOLDER = r"E:\桌面\project\ScaleDataset_eval\Model_result\biggermodel"
OUTPUT_SUFFIX = ""

# 定义特殊类别的前缀列表
SPECIAL_CATEGORIES = [
    "amps-ammeter",
    "angles-protractor",
    "ph-phmeter",
    "pressure-sphygmometer",
    "sound-intensitymeter",
    "speed-anemometer",
    "volts-voltmeter",
    "volume-testtube",
    "weight-steelyard",
    "weight-bodyweigh"
]

def processData(input_file, output_file):
    # 读取JSONL文件
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误发生在第 {line_num} 行:")
                    print(f"错误行内容: {line}")
                    print(f"错误详情: {str(e)}")
                    raise
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        raise
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        raise
    
    # 初始化计数器
    main_category_count = defaultdict(int)  # 主类别问题数量
    sub_category_count = defaultdict(int)   # 子类别问题数量
    main_category_correct = defaultdict(int)  # 主类别正确数量
    sub_category_correct = defaultdict(int)   # 子类别正确数量
    
    # special类别的计数器
    special_count = 0
    special_correct = 0
    
    # 每条数据作为一组处理
    for item in data:
        # 获取图片名称
        image_name = item['image']
        
        # 检查是否属于special类别
        is_special = False
        for special_prefix in SPECIAL_CATEGORIES:
            if image_name.startswith(special_prefix):
                is_special = True
                special_count += 1
                if item['result'] == 1:
                    special_correct += 1
                break
        
        # 如果不是special类别，才计入主类别和子类别
        if not is_special:
            main_category = image_name.split('-')[0]
            sub_category = image_name.split('-')[1]
            
            # 更新计数
            main_category_count[main_category] += 1
            sub_category_count[sub_category] += 1
            
            # 检查结果是否正确
            if item['result'] == 1:
                main_category_correct[main_category] += 1
                sub_category_correct[sub_category] += 1
    
    # 计算总体准确率
    total_questions = len(data)
    total_correct = sum(1 for item in data if item['result'] == 1)
    total_acc = total_correct / total_questions if total_questions > 0 else 0
    
    # 计算special类别的准确率
    special_acc = special_correct / special_count if special_count > 0 else 0
    
    # 写入结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"总体准确率: {total_acc:.4f} ({total_correct}/{total_questions})\n\n")
        
        # 输出special类别的准确率
        f.write(f"Special类别准确率: {special_acc:.4f} ({special_correct}/{special_count})\n\n")
        
        # 按主类别分组输出
        for main_category in sorted(main_category_count.keys()):
            f.write(f"主类别\n{main_category}: {main_category_correct[main_category]/main_category_count[main_category]:.4f} ({main_category_correct[main_category]}/{main_category_count[main_category]})\n")
            
            # 查找对应的子类别
            for sub_category in sorted(sub_category_count.keys()):
                # 检查这个子类别是否属于当前主类别
                if any(f"{main_category}-{sub_category}" in item['image'] for item in data if not any(item['image'].startswith(sp) for sp in SPECIAL_CATEGORIES)):
                    f.write(f"子类别：\n{sub_category}: {sub_category_correct[sub_category]/sub_category_count[sub_category]:.4f} ({sub_category_correct[sub_category]}/{sub_category_count[sub_category]})\n\n")

def processFolder(input_folder, output_folder, output_suffix):
    """
    处理文件夹中的所有jsonl文件
    
    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    output_suffix: 输出文件名后缀
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 获取所有jsonl文件
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"错误: 在 {input_folder} 中未找到jsonl文件")
        return
    
    # 处理每个文件
    for jsonl_file in jsonl_files:
        input_path = os.path.join(input_folder, jsonl_file)
        
        # 从文件名中提取完整的文件名（去掉.jsonl后缀）
        model_name = os.path.splitext(jsonl_file)[0]
        output_file_name = f"{model_name}{output_suffix}.txt"
        output_path = os.path.join(output_folder, output_file_name)
        
        print(f"处理文件: {jsonl_file}")
        print(f"输出结果到: {output_file_name}")
        
        try:
            processData(input_path, output_path)
            print(f"成功处理 {jsonl_file}")
        except Exception as e:
            print(f"处理 {jsonl_file} 时出错: {str(e)}")

if __name__ == "__main__":
    processFolder(INPUT_FOLDER, OUTPUT_FOLDER, OUTPUT_SUFFIX)
    print(f"所有文件处理完成，结果保存在 {OUTPUT_FOLDER} 文件夹中")
