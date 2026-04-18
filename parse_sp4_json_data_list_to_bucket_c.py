import json
from collections import defaultdict

def create_bucket_config(json_data):
    """
    从原始数据列表中解析并创建 bucket_config。

    Args:
        json_data (list): 从 JSON 文件加载的列表数据。

    Returns:
        dict: 格式化后的 bucket_config 字典。
    """
    # 使用 defaultdict 来方便地存储每个 frame 对应的最大 batch_size
    # 结构: {frames: max_batch_size}
    max_batch_sizes = defaultdict(int)

    # 假设我们只关心 1280x720 的分辨率
    target_h, target_w = 1280.0, 720.0

    # 遍历 JSON 数据中的每一条记录
    for record in json_data:
        # 解包五元组
        batch_size, frames, _, h, w = record

        # 检查是否是我们关心的分辨率
        if h == target_h and w == target_w:
            # 将 frames 和 batch_size 转为整数处理
            frames_int = int(frames)
            bs_int = int(batch_size)

            # 更新当前 frames 对应的最大 batch_size
            if bs_int > max_batch_sizes[frames_int]:
                max_batch_sizes[frames_int] = bs_int

    # 准备最终的输出字典
    # 按照你提供的格式，value是一个元组，第一个元素固定为1.0
    final_bucket = {}
    # 对 frames 进行排序，以获得有序的输出
    for frames in sorted(max_batch_sizes.keys()):
        max_bs = max_batch_sizes[frames]
        final_bucket[frames] = (1.0, max_bs)
        
    # 最终的完整配置结构
    bucket_config = {
        "960px": final_bucket
    }

    return bucket_config

def format_config_for_py_file(config):
    """
    将字典格式化为 python 文件中的字符串。
    """
    output_str = "bucket_config = {\n"
    for key, inner_dict in config.items():
        output_str += f'    "{key}": {{\n'
        # 计算 key 的最大长度以对齐
        items = list(inner_dict.items())
        for i, (frame, value) in enumerate(items):
            # 格式化元组
            value_str = f"({value[0]}, {value[1]})"
            output_str += f"        {frame}: {value_str},\n"
        output_str += "    }\n"
    output_str += "}\n"
    return output_str

# --- 使用示例 ---
# 假设这是从你的JSON文件 'your_data.json' 加载的数据
# 注意：此处的示例数据是模拟的，你需要用你自己的完整数据
with open('reusable_data_list_sp4.json', 'r') as f:
          json_list_data = json.load(f)

# json_list_data = [
#     # 模拟数据，以产生与你期望输出一致的结果
#     [26.0, 5.0, 3, 1280.0, 720.0], [10.0, 5.0, 3, 1280.0, 720.0],
#     [14.0, 9.0, 3, 1280.0, 720.0], [1.0, 9.0, 3, 1280.0, 720.0],
#     [10.0, 13.0, 3, 1280.0, 720.0],
#     [7.0, 17.0, 3, 1280.0, 720.0],
#     [6.0, 21.0, 3, 1280.0, 720.0],
#     [5.0, 25.0, 3, 1280.0, 720.0],
#     [4.0, 29.0, 3, 1280.0, 720.0],
#     [4.0, 33.0, 3, 1280.0, 720.0], [3.0, 33.0, 3, 1280.0, 720.0],
#     [3.0, 37.0, 3, 1280.0, 720.0],
#     [3.0, 41.0, 3, 1280.0, 720.0], [1.0, 41.0, 3, 1280.0, 720.0]
# ]

# 1. 解析数据并生成配置字典
generated_config = create_bucket_config(json_list_data)

# 2. 将字典格式化为字符串
config_string = format_config_for_py_file(generated_config)

output_python_file = "bucket_config_for_wan_sp4.py"
try:
    with open(output_python_file, 'w', encoding='utf-8') as f:
        f.write(config_string)
    print(f"配置已成功保存到文件: '{output_python_file}'")
except IOError as e:
    print(f"错误: 无法写入文件 '{output_python_file}'. 错误信息: {e}")


# 5. 同时在控制台打印最终结果
print("\n--- 生成内容预览 ---")
print(config_string)