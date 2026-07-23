# sudo docker run -it -v /data01/yfq/code/dx_train:/workspace/dx_train -v /root:/nv_data -v /root/datasets/Text2Video/data:/workspace/data --ipc=host --gpus all e608e0bd9cdb bash
import os
import sys
import cv2
import math
import json
import pandas as pd
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import hashlib
import torch
from PIL import Image
from tqdm import tqdm

import pdb


from vast.datasets import Dataset, load_dataset, utils
from teleai_data_tool.datasets.lmdb_dataset import LmdbDataset
from teleai_data_tool.datasets.pkl_dataset import PklDataset

#################################


def draw_body(canvas, points, scores, kpt_thr=0.4, stickwidth=2, r=2):
    colors = [
        [255, 0, 0],  # 0
        [0, 255, 0],  # 1
        [0, 0, 255],  # 2
        [255, 0, 255],  # 3
        [255, 255, 0],  # 4
        [85, 255, 0],  # 5
        [0, 75, 255],  # 6
        [0, 255, 85],  # 7
        [0, 255, 170],  # 8
        [170, 0, 255],  # 9
        [85, 0, 255],  # 10
        [0, 85, 255],  # 11
        [0, 255, 255],  # 12
        [85, 0, 255],  # 13
        [170, 0, 255],  # 14
        [255, 0, 255],  # 15
        [255, 0, 170],  # 16
        [255, 0, 85],  # 17
    ]
    connetions = [
        [17, 0],
        [0, 1],
        [0, 2],
        [2, 4],
        [1, 3],
        [17, 6],
        [6, 8],
        [8, 10],
        [17, 5],
        [5, 7],
        [7, 9],
        [17, 12],
        [12, 14],
        [14, 16],
        [17, 11],
        [11, 13],
        [13, 15],
    ]
    connection_colors = [
        [255, 0, 0],  # 0
        [0, 255, 0],  # 1
        [0, 0, 255],  # 2
        [255, 255, 0],  # 3
        [255, 0, 255],  # 4
        [0, 255, 0],  # 5
        [0, 85, 255],  # 6
        [255, 175, 0],  # 7
        [0, 0, 255],  ## 8
        [255, 85, 0],  # 9
        [0, 255, 85],  # 10
        [255, 0, 255],  # 11
        [255, 0, 0],  # 12
        [0, 175, 255],  # 13
        [255, 255, 0],  # 14
        [0, 0, 255],  # 15
        [0, 255, 0],  # 16
    ]
    if len(points) != 18:
        #  add neck point
        points.append(
            [(points[5][0] + points[6][0]) / 2, (points[5][1] + points[6][1]) / 2]
        )
        scores.append((scores[5] + scores[6]) / 2)

    # draw point
    for i in range(len(points)):
        score = scores[i]
        if i == 15 or i == 16:
            if score < 0.6:
                continue
        if score < kpt_thr:
            continue
        x, y = points[i][0:2]
        x, y = int(x), int(y)
        cv2.circle(canvas, (x, y), r, colors[i], thickness=-1)
        # cv2.putText(canvas, i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    # draw line
    for i in range(len(connetions)):
        i = 16 - i
        point1_idx, point2_idx = connetions[i][0:2]

        if point2_idx == 15 or point2_idx == 16:
            if scores[point2_idx] < 0.6:
                continue

        if scores[point1_idx] < kpt_thr or scores[point2_idx] < kpt_thr:
            continue

        point1 = points[point1_idx]
        point2 = points[point2_idx]
        Y = [point2[0], point1[0]]
        X = [point2[1], point1[1]]
        mX = int(np.mean(X))
        mY = int(np.mean(Y))
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, connection_colors[i])
        # cv2.putText(canvas, i)+connection_colors[i]), (mY, mX), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    return canvas


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def get_pose_frames(skeleton_data, H, W, H_out=None, W_out=None):
    pose_frames = []
    frame_num = len(skeleton_data["instance_info"])  # 视频帧数

    # 先保存每一帧的所有框
    all_boxes = dict()
    for frame_number in range(frame_num):
        person_num_in_frame = len(
            skeleton_data["instance_info"][frame_number]["instances"]
        )
        all_boxes[frame_number] = []
        for person_idx in range(person_num_in_frame):
            bbox = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["bbox"]
            all_boxes[frame_number].append(bbox[0])

    for frame_number in range(frame_num):
        person_num_in_frame = len(
            skeleton_data["instance_info"][frame_number]["instances"]
        )
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        curr_boxes, prev_boxes, next_boxes = None, None, None
        if frame_number > 0 and frame_number < frame_num - 1:
            curr_boxes = all_boxes[frame_number]
            prev_boxes = all_boxes[frame_number - 1]
            next_boxes = all_boxes[frame_number + 1]

        threshold = 0.5  # IoU 阈值
        for person_idx in range(person_num_in_frame):
            points = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["keypoints"]
            scores = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["keypoint_scores"]

            curr_box = all_boxes[frame_number][person_idx]
            overlap_prev, overlap_next = True, True
            # 过滤掉小于1/5 H的框
            if curr_box[3] - curr_box[1] < H / 5:
                continue
            if prev_boxes is not None:
                # 检查与前一帧的框的重合度
                overlap_prev = any(
                    iou(curr_box, prev_box) > threshold for prev_box in prev_boxes
                )
            if next_boxes is not None:
                # 检查与后一帧的框的重合度
                overlap_next = any(
                    iou(curr_box, next_box) > threshold for next_box in next_boxes
                )
            if not overlap_prev and not overlap_next:
                continue

            canvas = draw_body(
                canvas,
                points,
                scores,
                kpt_thr=0.1,
                stickwidth=math.floor(W / 350),
                r=math.floor(W / 450),
            )
            # canvas = draw_body(canvas, points, scores, kpt_thr=0.001,stickwidth=4,r=4)
        if H_out is not None:
            img_pil = Image.fromarray(canvas)
            img_resized = img_pil.resize((W_out, H_out), Image.Resampling.LANCZOS)
            canvas = np.array(img_resized)
        pose_frames.append(canvas)

    pose_frames = np.array(pose_frames)
    return vframes


def get_pose_frames_new(skeleton_data, H, W, frame_num=None, H_out=None, W_out=None):
    pose_frames = []
    # frame_num = len(skeleton_data['instance_info']) # 视频帧数
    person_id_list = list(skeleton_data.keys())
    if len(person_id_list) > 0:
        if frame_num > len(skeleton_data[person_id_list[0]]):
            frame_num = len(skeleton_data[person_id_list[0]])

    for frame_number in range(frame_num):
        # person_num_in_frame = len(skeleton_data["instance_info"][frame_number]["instances"])
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        for person_idx in range(len(person_id_list)):
            person_id = person_id_list[person_idx]
            pose_dict = skeleton_data[person_id][frame_number]

            # 只绘制有pose的人
            if isinstance(pose_dict, dict):
                points = pose_dict["keypoints"]
                scores = pose_dict["keypoint_scores"]
                curr_box = pose_dict["bbox"][0]
            else:
                continue
            # 过滤掉小于1/4 H的框
            if curr_box[3] - curr_box[1] < H / 4:
                continue

            canvas = draw_body(
                canvas,
                points,
                scores,
                kpt_thr=0.1,
                stickwidth=math.floor(W / 350),
                r=math.floor(W / 450),
            )

        if H_out is not None:
            img_pil = Image.fromarray(canvas)
            img_resized = img_pil.resize((W_out, H_out), Image.Resampling.LANCZOS)
            canvas = np.array(img_resized)

        pose_frames.append(canvas)

    pose_frames = np.array(pose_frames)
    return vframes


def string_to_rgb(color_name):
    """
    将字符串映射为 RGB 颜色。

    参数:
    color_name (str): 要映射的字符串。

    返回:
    tuple: 包含 (R, G, B) 值的元组，范围从 0 到 255。
    """
    # 使用 hashlib 创建哈希对象
    hash_object = hashlib.md5(color_name.encode())
    # 获取 16 进制的哈希值
    hex_dig = hash_object.hexdigest()
    # 将哈希值映射到 RGB 颜色空间
    r = int(hex_dig[:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    # 确保 RGB 值在 0-255 范围内
    r = r % 256
    g = g % 256
    b = b % 256
    return (r, g, b)


def get_segment_frames(segment, height, width, num_frames):
    imgs = []
    for idx_img in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for key, value in segment.items():
            if key.split("#")[0].lower() not in [
                "man",
                "woman",
                "boy",
                "girl",
                "person",
                "helmet",
            ]:
                continue
            if (
                idx_img < len(segment[key])
                and segment[key][idx_img] != []
                and os.path.exists(segment[key][idx_img])
            ):
                dense_matrix = scipy.sparse.load_npz(segment[key][idx_img]).toarray()
                if np.sum(dense_matrix) < (height * width / 150.0):
                    continue
                img[dense_matrix, :] += np.array(
                    string_to_rgb(key.split("#")[0]), dtype=np.uint8
                )
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def get_box_frames(track, height, width, num_frames):
    imgs = []
    for idx_img in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for key, value in track.items():
            if idx_img < len(value) and value[idx_img] != []:
                boxes = value[idx_img]
                if (
                    len(boxes) > 0
                    and (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
                    < height * width / 100
                ):
                    continue
                img = cv2.rectangle(
                    img,
                    (boxes[0], boxes[1]),
                    (boxes[2], boxes[3]),
                    string_to_rgb(key.split("#")[0]),
                    thickness=3,
                )
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def check(sub_data):
    try:
        if not os.path.exists(sub_data["path"]):
            return False
        cap = cv2.VideoCapture(sub_data["path"])

        if not cap.isOpened():
            return False
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if sub_data["num_frames"] != frame_count:
            return False
        if sub_data["width"] != width:
            return False
        if sub_data["height"] != height:
            return False

        segment = json.loads(sub_data["segment"])
        track = json.loads(sub_data["track"])
        pose = json.loads(sub_data["pose"])

        for k, v in segment.items():
            if len(v) != frame_count:
                return False
        for k, v in track.items():
            if len(v) != frame_count:
                return False
        for k, v in pose.items():
            if len(v) != frame_count:
                return False

        return True
    except Exception as e:
        return False


def convert_to_giga(dx_pose_dict):
    if len(dx_pose_dict) == 0:
        return [], [], []

    box_all = []
    score_all = []
    pose_all = []

    for k, v in dx_pose_dict.items():
        num_frames = len(v)
        break

    for i in range(num_frames):
        per_frame_keypoints = []
        per_frame_keypoint_scores = []
        pre_frame_bbox = []
        per_frame_bbox_score = []
        # hands = []
        # faces = []
        per_frame_subset = []
        subset_0 = np.arange(0, 18, dtype=np.int32)

        for person_id in dx_pose_dict.keys():
            person = dx_pose_dict[person_id]
            pose = person[i]

            if pose is not None and isinstance(pose, dict):
                per_frame_keypoints.append(pose["keypoints"])
                per_frame_keypoint_scores.append(pose["keypoint_scores"])
                pre_frame_bbox.append(pose["bbox"])
                per_frame_bbox_score.append(pose["bbox_score"])
                # hands.append(pose["hands"])
                # faces.append(pose["faces"])
                per_frame_subset.append(subset_0)
                subset_0 = subset_0 + 18

        if len(per_frame_keypoints) > 0:  # 可能这一帧没有pose
            per_frame_keypoints = np.concatenate(per_frame_keypoints, axis=0)  # (n, 18)
            per_frame_subset = np.array(per_frame_subset)  # (n, 18)
            pre_frame_bbox = np.concatenate(pre_frame_bbox, axis=0)  # (n, 4)
            per_frame_bbox_score = np.array(per_frame_bbox_score)  # (n,)
            per_frame_keypoint_scores = np.array(per_frame_keypoint_scores)  # (n,)
            # hands = np.concatenate(hands, axis=0)
            # faces = np.array(faces)
        else:
            per_frame_keypoints = np.array([])  # (0,)
            per_frame_subset = np.array([])  # (0,)
            pre_frame_bbox = np.array([])
            per_frame_bbox_score = np.array([])
            per_frame_keypoint_scores = np.array([])

        score_all.append(per_frame_keypoint_scores)
        box_all.append(pre_frame_bbox)
        pose_all.append(
            {
                "bodies": {
                    "candidate": per_frame_keypoints,
                    "subset": per_frame_subset,
                },
                # "hands": hands,
                # "faces": faces,
            }
        )

    return pose_all, box_all, score_all


class ImageConverter:
    def __init__(self, save_path, save_version="v0.0.1"):
        os.makedirs(os.path.join(save_path, save_version), exist_ok=True)
        self.save_path = save_path
        self.save_version = save_version
        self.save_label_path = os.path.join(save_path, save_version, "labels")
        self.save_image_path = os.path.join(save_path, save_version, "images")
        self.save_mask_path = os.path.join(save_path, save_version, "mask")
        self.save_pose_path = os.path.join(save_path, save_version, "pose")
        self.save_bbox_path = os.path.join(save_path, save_version, "bbox")

    def __call__(self, csv_data):
        label_writer = PklWriter(self.save_label_path)
        image_writer = LmdbWriter(self.save_image_path)
        mask_writer = LmdbWriter(self.save_mask_path)
        pose_writer = LmdbWriter(self.save_pose_path)
        bbox_writer = LmdbWriter(self.save_bbox_path)

        global_idx = 0
        # for idx in tqdm(range(len(image_paths))):
        for idx in tqdm(range(len(csv_data))):
            try:
                sub_data = csv_data.iloc[idx]
                if not check(sub_data):
                    continue

                video_length = sub_data["num_frames"]
                video_height = sub_data["height"]
                video_width = sub_data["width"]

                # pose_path = sub_data["pose"]
                # if pose_path.endswith((".json")):
                #     with open(pose_path, 'r', encoding='utf-8') as file:
                #         skeleton_data = json.load(file)
                #         vframes_pose = get_pose_frames(skeleton_data, video_height, video_width)
                # else:
                #     skeleton_data = json.loads(pose_path)
                #     vframes_pose = get_pose_frames_new(skeleton_data, video_height, video_width, video_length)

                # segment = json.loads(sub_data['segment'])
                # track = json.loads(sub_data["track"])

                # pose = get_pose_frames_new(skeleton_data, video_height, video_width, video_length)
                # mask = get_segment_frames(segment, track, video_height, video_width, video_length)
                # bbox = get_box_frames(track, video_height, video_width, video_length)

                # debug useage
                # from read_video import read_video
                # vframes, _ = read_video(sub_data["path"], backend="av")
                # cv2.imwrite("test_img.jpg", vframes[5].numpy().transpose(1,2,0)[:,:,::-1])
                # cv2.imwrite("test_bbox.jpg", bbox[5].transpose(1,2,0).astype(np.int8))
                # cv2.imwrite("test_mask.jpg", mask[5].transpose(1,2,0).astype(np.int8))
                # cv2.imwrite("test_pose.jpg", pose[5].transpose(1,2,0).astype(np.int8))

                image_writer.write_video(global_idx, sub_data["path"])
                # pose_writer.write_numpy(global_idx, pose)
                # bbox_writer.write_numpy(global_idx, bbox)
                # mask_writer.write_numpy(global_idx, mask)
                label_writer.write_dict(
                    {
                        "data_index": global_idx,
                        "video_length": sub_data["num_frames"],
                        "video_height": sub_data["height"],
                        "video_width": sub_data["width"],
                        "prompt": sub_data["text"],
                        # 'mask': sub_data["segment"],
                        # 'bbox': sub_data["track"],
                        # 'pose': sub_data["pose"],
                    }
                )
                global_idx += 1
            except Exception as e:
                import traceback

                print(e)
                traceback.print_exc()

        print("total cnt:", global_idx)
        label_writer.write_config()
        image_writer.write_config()
        # mask_writer.write_config()
        # pose_writer.write_config()
        # bbox_writer.write_config()

        label_writer.close()
        image_writer.close()
        # mask_writer.close()
        # pose_writer.close()
        # bbox_writer.close()

        label_dataset = load_dataset(self.save_label_path)
        image_dataset = load_dataset(self.save_image_path)
        # mask_dataset = load_dataset(self.save_mask_path)
        # pose_dataset = load_dataset(self.save_pose_path)
        # bbox_dataset = load_dataset(self.save_bbox_path)

        dataset = Dataset(
            [
                label_dataset,
                image_dataset,
            ]
        )
        #    mask_dataset, bbox_dataset])

        dataset.save(os.path.join(self.save_path, self.save_version))


class ImageConverter2:
    def __init__(self, save_path, save_version="v0.0.1"):
        os.makedirs(os.path.join(save_path, save_version), exist_ok=True)
        self.save_path = save_path
        self.save_version = save_version
        self.save_label_path = os.path.join(save_path, save_version, "labels")
        self.save_image_path = os.path.join(save_path, save_version, "images")
        self.save_pose_path = os.path.join(save_path, save_version, "pose")

    def __call__(self, csv_data):
        label_writer = PklWriter(self.save_label_path)
        image_writer = LmdbWriter(self.save_image_path)
        pose_writer = LmdbWriter(self.save_pose_path)

        global_idx = 0
        # for idx in tqdm(range(len(image_paths))):
        for idx in tqdm(range(len(csv_data))):
            try:
                sub_data = csv_data.iloc[idx]
                if not check(sub_data):
                    continue

                video_length = sub_data["num_frames"]
                video_height = sub_data["height"]
                video_width = sub_data["width"]

                image_writer.write_video(global_idx, sub_data["path"])
                pose = json.loads(sub_data["pose"])
                pose_all, box_all, score_all = convert_to_giga(pose)
                new_pose = {
                    "poses": pose_all,
                    "poses_scores": score_all,
                    "boxes": box_all,
                }

                pose_writer.write_dict(global_idx, new_pose)
                label_writer.write_dict(
                    {
                        "data_index": global_idx,
                        "video_length": sub_data["num_frames"],
                        "video_height": sub_data["height"],
                        "video_width": sub_data["width"],
                        "prompt": sub_data["text"],
                    }
                )
                global_idx += 1
            except Exception as e:
                import traceback

                print(e)
                traceback.print_exc()

        print("total cnt:", global_idx)
        label_writer.write_config()
        image_writer.write_config()
        pose_writer.write_config()

        label_writer.close()
        image_writer.close()
        pose_writer.close()

        label_dataset = load_dataset(self.save_label_path)
        image_dataset = load_dataset(self.save_image_path)
        pose_dataset = load_dataset(self.save_pose_path)

        dataset = Dataset([label_dataset, image_dataset, pose_dataset])
        dataset.save(os.path.join(self.save_path, self.save_version))


def main():
    # csv_paths = [
    #     '/nv_data/Text2Video/yfq/data/results_csvs/fight_filter_csv_quchong_1028.csv',
    #     '/nv_data/Text2Video/data/result_csv/guangdian/66_new.csv',
    #     '/nv_data/Text2Video/data/result_csv/guangdian/67_new.csv',
    # ]
    csv_paths = [
        # '/nv_data/Text2Video/yfq/data/results_csvs/fight_filter_csv_quchong_1028.csv',
        "/nv_data/Text2Video/data/result_csv/guangdian/66_new.csv",
        "/nv_data/Text2Video/data/result_csv/guangdian/67_new.csv",
    ]
    csv_data = pd.concat(
        [pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True
    )
    # csv_data = csv_data.sample(20, ignore_index=True)
    save_path = "/nv_data/dj/users/yfq/datasets/giga_guangdian_data"
    ImageConverter2(save_path=save_path, save_version="v0.0.1")(csv_data)


if __name__ == "__main__":
    main()
