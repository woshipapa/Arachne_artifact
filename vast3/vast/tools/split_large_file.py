import os
import re


def split_file(file_path, chunk_size=100 * 1024 * 1024):
    """
    将文件拆分成多个小于1GB的文件。

    :param file_path: 要拆分的文件路径
    :param chunk_size: 每个拆分文件的最大大小，默认为1GB
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。")
        return

    file_name, file_ext = os.path.splitext(file_path)
    chunk_number = 1

    with open(file_path, "rb") as file:
        chunk = file.read(chunk_size)
        while chunk:
            chunk_file_path = f"{file_name}_part{chunk_number}{file_ext}"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk)
            print(f"已创建文件：{chunk_file_path}")
            chunk_number += 1
            chunk = file.read(chunk_size)


def merge_files(base_file_path, output_file_path):
    """
    合并拆分后的文件。

    :param base_file_path: 原始文件的基本路径（不包含_partX部分）
    :param output_file_path: 合并后的输出文件路径
    """
    # 获取所有拆分文件
    pattern = re.compile(rf"{re.escape(base_file_path)}_part(\d+)")
    files = [f for f in os.listdir(".") if pattern.match(f)]

    # 按数字顺序排序文件
    files.sort(key=lambda x: int(pattern.match(x).group(1)))

    with open(output_file_path, "wb") as output_file:
        for file_name in files:
            with open(file_name, "rb") as input_file:
                output_file.write(input_file.read())
            print(f"已合并文件：{file_name}")


if __name__ == "__main__":
    split_file(
        "/root/users/xinze.chen/experiments/cogvideox/cogvideox_f49_720p_v1/models/checkpoint_epoch_3_step_15000/transformer/diffusion_pytorch_model.bin"
    )
    # merge_files('path_to_your_base_file', 'path_to_your_merged_file.bin')
