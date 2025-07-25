import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore")
input_folder = "your_input_folder"
output_folder = "your_output_folder"
result_file = output_folder + "result.txt"
all_correct = 0


def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    if line.strip():  
                        tmp = json.loads(line.strip())
                        if tmp['result'] == 1:
                            tmp['output'] = tmp['answer']
                        data.append(tmp)
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse file {os.path.basename(file_path)} at line {line_number}: {e}")
                    print(f"Problem content: {line[:50]}...")
    except Exception as e:
        print(f"Error: Failed to read file {file_path}: {e}")
    return data


def main():
    file_list = os.listdir(input_folder)
    
    
    # Create result output file
    with open(result_file, 'w', encoding='utf-8') as result_out:
        for test_file_path_item in file_list:
            test_file_path = input_folder + "\\" + test_file_path_item

            print("------------------------------------" + test_file_path_item + "-----------------------------------")
            result_out.write("------------------------------------" + test_file_path_item + "-----------------------------------\n")

            data_list = load_jsonl(test_file_path)
            print(len(data_list))
            result_out.write(f"Sample count: {len(data_list)}\n")
            
            A_pre_num, B_pre_num, C_pre_num, D_pre_num, \
                A_label_num, B_label_num, C_label_num, D_label_num, \
                A_true_num, B_true_num, C_true_num, D_true_num = \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            for data in data_list:
                if data['output'] in "A" or "A" in data['output']:
                    A_pre_num += 1
                elif data['output'] in "B" or "B" in data['output']:
                    B_pre_num += 1
                elif data['output'] in "C" or "C" in data['output']:
                    C_pre_num += 1
                elif data['output'] in "D" or "D" in data['output']:
                    D_pre_num += 1

                if data['answer'] in "A" or "A" in data['answer']:
                    A_label_num += 1
                elif data['answer'] in "B" or "B" in data['answer']:
                    B_label_num += 1
                elif data['answer'] in "C" or "C" in data['answer']:
                    C_label_num += 1
                elif data['answer'] in "D" or "D" in data['answer']:
                    D_label_num += 1

                if data['result'] == 1:
                    if data['answer'] == "A":
                        A_true_num += 1
                    elif data['answer'] == "B":
                        B_true_num += 1
                    elif data['answer'] == "C":
                        C_true_num += 1
                    elif data['answer'] == "D":
                        D_true_num += 1

            # print(A_true_num,A_pre_num,A_label_num,B_true_num,B_pre_num,B_label_num,C_true_num,C_pre_num,C_label_num,D_true_num,D_pre_num,D_label_num)
            result_out.write(f"A_true_num: {A_true_num}, A_pre_num: {A_pre_num}, A_label_num: {A_label_num}\n")
            result_out.write(f"B_true_num: {B_true_num}, B_pre_num: {B_pre_num}, B_label_num: {B_label_num}\n")
            result_out.write(f"C_true_num: {C_true_num}, C_pre_num: {C_pre_num}, C_label_num: {C_label_num}\n")
            result_out.write(f"D_true_num: {D_true_num}, D_pre_num: {D_pre_num}, D_label_num: {D_label_num}\n")
            
            # Avoid division by zero errors
            A_precision = A_true_num / A_pre_num if A_pre_num > 0 else 0
            B_precision = B_true_num / B_pre_num if B_pre_num > 0 else 0
            C_precision = C_true_num / C_pre_num if C_pre_num > 0 else 0
            D_precision = D_true_num / D_pre_num if D_pre_num > 0 else 0
            
            A_recall = A_true_num / A_label_num if A_label_num > 0 else 0
            B_recall = B_true_num / B_label_num if B_label_num > 0 else 0
            C_recall = C_true_num / C_label_num if C_label_num > 0 else 0
            D_recall = D_true_num / D_label_num if D_label_num > 0 else 0
            
            p = (A_precision + B_precision + C_precision + D_precision) / 4
            r = (A_recall + B_recall + C_recall + D_recall) / 4
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            # Accuracy
            correct_count = sum(1 for data in data_list if data['output'] == data['answer'])
            acc = correct_count / len(data_list) if len(data_list) > 0 else 0

            result_out.write(f"Precision: {p:.4f}\n")
            result_out.write(f"Recall: {r:.4f}\n")
            result_out.write(f"F1 Score: {f1:.4f}\n")
            result_out.write(f"Accuracy: {acc:.4f}\n\n")
    
    print(f"Evaluation results saved to file: {result_file}")


if __name__ == "__main__":
    main()