import os
import pandas as pd
import json
import re

def process_schedules_to_json(directory_path, output_filename="iteration_to_makespan.json"):
    """
    扫描指定目录下的CSV trace文件，提取iteration和makespan，并生成一个JSON文件。

    参数:
    directory_path (str): 包含CSV文件的目录路径。
    output_filename (str): 输出的JSON文件名。
    """
    # 初始化一个字典来存储结果
    results = {}

    # 编译正则表达式以提高效率
    # \d+ 匹配一个或多个数字, () 创建一个捕获组
    file_pattern = re.compile(r'schedule_(\d+)_trace\.csv')

    print(f"开始扫描目录: '{directory_path}'")

    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误：找不到目录 '{directory_path}'。请确认脚本的运行位置或修改路径。")
        return

    # 获取文件列表并按自然顺序排序文件名中的数字
    try:
        file_list = sorted(os.listdir(directory_path), key=lambda x: int(file_pattern.match(x).group(1)) if file_pattern.match(x) else 0)
    except (ValueError, AttributeError):
         # 如果文件名格式不统一，则使用简单排序
        file_list = sorted(os.listdir(directory_path))
        print("警告：部分文件名不符合预期格式，使用默认字母顺序排序。")
        
    processed_files = 0
    # 遍历目录中的所有文件
    for filename in file_list:
        match = file_pattern.match(filename)
        # 如果文件名符合我们的模式
        if match:
            try:
                # 从文件名中提取 iteration
                iteration = int(match.group(1))
                full_path = os.path.join(directory_path, filename)

                # 读取CSV文件并提取Makespan
                df = pd.read_csv(full_path)
                
                if not df.empty and 'Makespan' in df.columns:
                    # Makespan在整个文件中都是一样的，所以我们只取第一个
                    makespan_time = df['Makespan'].iloc[0]
                    
                    # 将结果存入字典
                    results[iteration] = makespan_time
                    processed_files += 1
                else:
                    print(f"警告: 在文件 {filename} 中找不到 'Makespan' 列或文件为空。")

            except Exception as e:
                print(f"处理文件 {filename} 时发生错误: {e}")
    
    if processed_files == 0:
        print("未找到任何符合格式 'schedule_{iteration}_trace.csv' 的文件。")
        return

    # 按iteration对字典进行排序以确保JSON输出的有序性
    sorted_results = dict(sorted(results.items()))

    # 将结果写入JSON文件
    try:
        with open(output_filename, 'w') as f:
            json.dump(sorted_results, f, indent=4)
        print(f"\n处理完成！共处理了 {processed_files} 个文件。")
        print(f"结果已成功保存到文件: {output_filename}")
    except Exception as e:
        print(f"\n保存JSON文件时发生错误: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    # <<<<<<<<<<<<<<<< 在这里定义您的CSV文件路径 >>>>>>>>>>>>>>>>
    target_directory = 't2v_flow/planner/generated_schedules/hunyuan/720p/53'
    
    process_schedules_to_json(target_directory)