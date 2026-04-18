import json
from collections import OrderedDict

def generate_bucket_config(oom_file='oom.json', output_file='bucket_config.py'):
    """
    根据 oom.json 文件生成 bucket_config.py 配置文件。
    使用 range 动态生成帧数桶。

    Args:
        oom_file (str): 输入的 oom json 文件路径。
        output_file (str): 输出的 bucket config python 文件路径。
    """
    try:
        with open(oom_file, 'r') as f:
            oom_data_str_keys = json.load(f)
        # 将json中的字符串键转换为整数键
        oom_data = {int(k): v for k, v in oom_data_str_keys.items()}
        print(f"成功读取并解析 {oom_file}")
    except FileNotFoundError:
        print(f"错误: 未找到 {oom_file} 文件。请确保该文件存在于当前目录。")
        return
    except json.JSONDecodeError:
        print(f"错误: {oom_file} 文件格式不正确，无法解析。")
        return

    # --- 修改部分 ---
    # 使用 range 动态生成帧数桶，从 5 到 129，步长为 4
    # range 的第二个参数是 'stop'，它本身不被包含在内，所以我们用 130
    frame_buckets = list(range(5, 130, 4))
    print(f"已动态生成帧数桶: 从 {frame_buckets[0]} 到 {frame_buckets[-1]}, 步长为 4")
    # --- 修改结束 ---

    new_bucket_config = OrderedDict()

    # 为每一个帧数桶计算最大的可用 batch_size
    for frame_count in frame_buckets:
        max_bs = 0
        # 遍历 oom_data 来寻找可以支持当前 frame_count 的最大 batch size
        for bs, max_frames in oom_data.items():
            if max_frames >= frame_count:
                if bs > max_bs:
                    max_bs = bs
        
        if max_bs > 0:
            new_bucket_config[frame_count] = (1.0, max_bs)
        else:
            print(f"警告: 对于 {frame_count} 帧, 未在 {oom_file} 中找到任何支持的 batch size。")

    # 准备写入文件的内容
    output_content = 'bucket_config = {\n'
    output_content += '    "960px": {\n'
    for frame, (val1, val2) in new_bucket_config.items():
        output_content += f'        {frame}: ({val1}, {val2}),\n'
    output_content += '    },\n'
    output_content += '}\n'

    # 将内容写入到 .py 文件
    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"成功生成配置文件: {output_file}")
    print("\n--- 生成的文件内容预览 ---")
    print(output_content)


if __name__ == '__main__':
    model = "hunyuan"
    resolution = "720p"
    generate_bucket_config(oom_file=f"t2v_flow/predictor/oom_thresholds/hunyuan/oom_thresholds_{model}_{resolution}_sp4.json",
                           output_file=f"bucket_config_for_{model}_{resolution}_sp4.py")