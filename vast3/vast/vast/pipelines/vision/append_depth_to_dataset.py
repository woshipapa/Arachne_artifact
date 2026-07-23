# ref: https://github.com/DepthAnything/Depth-Anything-V2
"""
usage:
export HF_HOME=/root/share/
"""

from transformers import pipeline
from PIL import Image
import os
import sys
import json
from tqdm import tqdm
import argparse
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from multiprocessing import Process
import multiprocessing as mp
from teleai_data_tool.datasets.pkl_dataset import PklDataset
from vast.datasets import load_dataset


def split_into_n(lst, n):
    """
    将列表lst尽可能平均地分成n个子列表
    :param lst: 输入的列表
    :param n: 要分成的子列表数量
    :return: 包含n个子列表的列表
    """
    base_length = len(lst) // n
    remainder = len(lst) % n

    # 初始化子列表列表
    divided_lists = []

    # 遍历原列表并分配元素到子列表
    start_index = 0
    for i in range(n):
        # 计算当前子列表应该包含的元素数量
        end_index = start_index + base_length + (1 if i < remainder else 0)
        # 将原列表中的元素切片并添加到子列表列表中
        divided_lists.append(lst[start_index:end_index])
        # 更新下一个子列表的起始索引
        start_index = end_index

    return divided_lists


def yeild_data(video):
    for color_image in video:
        color_image = color_image.asnumpy()
        color_image = Image.fromarray(color_image)
        yield color_image


def run(dataset, global_idx_list, device, save_dir):
    model = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=device,
        batch_size=32,
    )
    for idx in tqdm(global_idx_list):
        # data_dict = dataset._get_data(0)
        data_dict = dataset.datasets[0][idx]
        for i in range(1, len(dataset.datasets)):
            data_dict.update(dataset.datasets[i][idx])
            # 只需要读到video就行
            if "video" in data_dict.keys():
                break
        video_path = data_dict["tos_addr"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        preds_path = os.path.join(save_dir, video_name + ".npz")
        if os.path.exists(preds_path):
            continue

        preds = []
        for pred in model(yeild_data(data_dict["video"])):
            preds.append(pred["depth"])

        preds = np.stack(preds)
        print(preds.shape)
        preds = csr_matrix(preds.reshape(-1, preds.shape[-1]))
        save_npz(preds_path, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset. valid path should have config.json",
    )
    parser.add_argument("--new_dataset_path", type=str, help="Path to the new dataset.")
    # dataset_path: /root/datasets/Text2Video/new_ppl_data/crawl_2024-11-15_2024-11-21/sport/bilibili/sport-t台走秀/v1.0/pack
    # new_dataset_path: /root/datasets/Text2Video/new_ppl_data/crawl_2024-11-15_2024-11-21/sport/bilibili/sport-t台走秀/v2.0/pack

    parser.add_argument("--user", type=str, help="your name")
    parser.add_argument(
        "--commit_message", type=str, help="what's new in this update.", default="None"
    )
    parser.add_argument("--num_process", type=int, default=8)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    dataset_length = len(dataset)

    # pack/depth/config.json
    # pack/depth/depth_npz/video_name.npz..
    pkl_data_path = os.path.join(args.new_dataset_path, "depth")
    save_dir = os.path.join(args.new_dataset_path, "depth", "depth_npz")
    if not os.path.exists(pkl_data_path):
        os.makedirs(pkl_data_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mp.set_start_method("spawn")
    idx_lists = split_into_n([i for i in range(dataset_length)], args.num_process)
    gpu_ids = [1, 2, 3, 4, 5, 6, 7]

    process_list = []
    for process_i in range(args.num_process):
        process = Process(
            target=run,
            args=(
                dataset,
                idx_lists[process_i],
                gpu_ids[process_i % len(gpu_ids)],
                save_dir,
            ),
        )
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()

    pkl_writer = PklWriter(pkl_data_path)
    print("Merging...")
    for idx in range(dataset_length):
        data_dict = dataset.datasets[0][idx]
        video_path = data_dict["tos_addr"]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        preds_path = os.path.join(save_dir, video_name + ".npz")

        if os.path.exists(preds_path):
            pkl_writer.write_dict(
                {"depth": "depth/depth_npz/{}.npz".format(video_name)}
            )  # os.path.join(root, depth_path)
        else:
            raise Exception()

    pkl_writer.write_config()
    pkl_writer.close()

    # cp config
    old_config_path = os.path.join(args.dataset_path, "config.json")
    json_data = json.load(open(old_config_path))

    new_config_path = []
    for config_path in json_data["config_paths"]:
        new_config_path.append(
            os.path.join(
                os.path.relpath(args.dataset_path, args.new_dataset_path), config_path
            )
        )
    new_config_path.append("depth/config.json")
    json_data["config_paths"] = new_config_path
    json_data["modified_user"] = args.user
    json_data["modified_message"] = args.commit_message

    json.dump(json_data, open(os.path.join(args.new_dataset_path, "config.json"), "w"))
