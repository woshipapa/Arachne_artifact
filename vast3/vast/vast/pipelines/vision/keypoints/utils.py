import os
import re

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image
import cv2
import math
import torch.nn.functional as F
import random


joint = {
    1: [0, 2, 5, 8, 11],
    0: [14, 15],
    14: [16],
    15: [17],
    2: [3],
    3: [4],
    5: [6],
    6: [7],
    8: [9],
    9: [10],
    11: [12],
    12: [13],
    # 7: [18],
    18: [19, 23, 27, 31, 35],
    19: [20],
    20: [21],
    21: [22],
    23: [24],
    24: [25],
    25: [26],
    27: [28],
    28: [29],
    29: [30],
    31: [32],
    32: [33],
    33: [34],
    35: [36],
    36: [37],
    37: [38],
    39: [40, 44, 48, 52, 56],
    40: [41],
    41: [42],
    42: [43],
    44: [45],
    45: [46],
    46: [47],
    48: [49],
    49: [50],
    50: [51],
    52: [53],
    53: [54],
    54: [55],
    56: [57],
    57: [58],
    58: [59],
}


def get_pose_frames(skeleton_data, H, W, H_out=None, W_out=None):
    pose_frames = []
    if "instance_info" in skeleton_data:
        frame_num = len(skeleton_data["instance_info"])  # 视频帧数
    else:
        frame_num = 0

    for frame_number in range(frame_num):
        person_num_in_frame = len(
            skeleton_data["instance_info"][frame_number]["instances"]
        )
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        for person_idx in range(person_num_in_frame):
            points = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["keypoints"]
            scores = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["keypoint_scores"]
            curr_box = skeleton_data["instance_info"][frame_number]["instances"][
                person_idx
            ]["bbox"][0]

            # 过滤掉小于1/4 H的框
            if curr_box[3] - curr_box[1] < H / 4:
                continue

            canvas = draw_body(
                canvas,
                points,
                scores,
                kpt_thr=0.3,
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
    return pose_frames


def get_pose_frames_infer(skeleton_data, H, W, frame_num=None, H_out=None, W_out=None):
    pose_frames = []
    # frame_num = len(skeleton_data['instance_info']) # 视频帧数
    person_id_list = list(skeleton_data.keys())
    if len(person_id_list) > 0:
        if frame_num > len(skeleton_data[person_id_list[0]]):
            frame_num = len(skeleton_data[person_id_list[0]])

    is_insert = False
    pre_points = None
    for frame_number in range(frame_num):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        for person_idx in range(len(person_id_list)):
            person_id = person_id_list[person_idx]
            pose_dict = skeleton_data[person_id][frame_number]

            hands_left = None

            # 只绘制有pose的人
            if isinstance(pose_dict, dict):
                points = pose_dict["keypoints"]
                scores = pose_dict["keypoint_scores"]
                if "bbx" in pose_dict:
                    curr_box = pose_dict["bbx"][0]
                else:
                    curr_box = pose_dict["bbox"][0]
                if "hands_left" in pose_dict:
                    hands_left = pose_dict["hands_left"]
                # hands_right = pose_dict['hands_right']
                # hands_left_score = pose_dict['hands_left_score']
                # hands_right_score = pose_dict['hands_right_score']
            else:
                continue
            # 过滤掉小于1/5 H的框
            if curr_box[3] - curr_box[1] < H / 5:
                continue

            # if person_id != "person#0":
            #     continue
            # print("person_id:",person_id)
            # print("points:",len(points),points[0])

            if frame_number == 0:
                pre_points = points
            # 遍历每个插值帧
            if frame_number > 1110 and frame_number < 10:
                insert_frames = 1
                for i in range(1, insert_frames + 1):
                    alpha = i / (insert_frames + 1)
                    # 计算插值点
                    interpolated_frame = []
                    for p1, p2 in zip(pre_points, points):
                        x_new = p1[0] + alpha * (p2[0] - p1[0])
                        y_new = p1[1] + alpha * (p2[1] - p1[1])
                        interpolated_frame.append([x_new, y_new])
                    canvas_temp = np.zeros(shape=(H, W, 3), dtype=np.uint8)
                    canvas_temp = draw_rtmpose(
                        canvas_temp,
                        points,
                        scores,
                        kpt_thr=0.3,
                        stickwidth=math.floor(W / 350),
                        r=math.floor(W / 450),
                    )
                    if H_out is not None:
                        img_pil = Image.fromarray(canvas_temp)
                        img_resized = img_pil.resize(
                            (W_out, H_out), Image.Resampling.LANCZOS
                        )
                        canvas_temp = np.array(img_resized)
                    pose_frames.append(canvas_temp)
                pre_points = points

            if hands_left is None:
                canvas = draw_body(
                    canvas,
                    points,
                    scores,
                    kpt_thr=0.01,
                    stickwidth=math.floor(W / 350),
                    r=math.floor(W / 450),
                )
            else:
                canvas = draw_rtmpose(
                    canvas,
                    points,
                    scores,
                    kpt_thr=0.38,
                    stickwidth=math.floor(W / 350),
                    r=math.floor(W / 450),
                )

        if H_out is not None:
            img_pil = Image.fromarray(canvas)
            img_resized = img_pil.resize((W_out, H_out), Image.Resampling.LANCZOS)
            canvas = np.array(img_resized)

        pose_frames.append(canvas)

    pose_frames = np.array(pose_frames)
    return pose_frames


def get_pose_frames_new(skeleton_data, H, W, frame_num=None, H_out=None, W_out=None):
    pose_frames = []
    # frame_num = len(skeleton_data['instance_info']) # 视频帧数
    person_id_list = list(skeleton_data.keys())
    if len(person_id_list) > 0:
        if frame_num > len(skeleton_data[person_id_list[0]]):
            frame_num = len(skeleton_data[person_id_list[0]])

    max_W, max_H, min_W, min_H = 0, 0, W, H
    for frame_number in range(frame_num):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        for person_idx in range(len(person_id_list)):
            person_id = person_id_list[person_idx]
            pose_dict = skeleton_data[person_id][frame_number]

            hands_left = None
            # 只绘制有pose的人
            if isinstance(pose_dict, dict):
                points = pose_dict["keypoints"]
                scores = pose_dict["keypoint_scores"]
                if "bbx" in pose_dict:
                    curr_box = pose_dict["bbx"][0]
                else:
                    curr_box = pose_dict["bbox"][0]
            else:
                continue
            # 过滤掉小于1/5 H的框
            if curr_box[3] - curr_box[1] < H / 5:
                continue
            # min_W = min(min_W,curr_box[0])
            # min_H = min(min_H,curr_box[1])
            # max_W = max(max_W,curr_box[2])
            # max_H = max(max_H,curr_box[3])

            if "hands_left" in pose_dict:
                canvas = draw_rtmpose(
                    canvas,
                    points,
                    scores,
                    kpt_thr=0.4,
                    stickwidth=math.floor(W / 350),
                    r=math.floor(W / 450),
                    pose_dict=pose_dict,
                )
            else:
                canvas = draw_body(
                    canvas,
                    points,
                    scores,
                    kpt_thr=0.01,
                    stickwidth=math.floor(W / 350),
                    r=math.floor(W / 450),
                )

        if H_out is not None:
            img_pil = Image.fromarray(canvas)
            img_resized = img_pil.resize((W_out, H_out), Image.Resampling.LANCZOS)
            canvas = np.array(img_resized)
        pose_frames.append(canvas)

    # print(max_W,max_H,min_W,min_H,W,H)
    pose_frames = np.array(pose_frames)
    # print(pose_frames.shape)

    # if not(max_W==W and max_H==H and min_W==0 and min_H==0):
    #     #scale and translation
    #     T, H, W, C = pose_frames.shape
    #     cropped_padded = np.zeros_like(pose_frames)
    #     random_value = random.uniform(0.8, 1.2)
    #     H1,W1 = int(H*random_value),int(W*random_value)
    #     pose_frames_ = torch.tensor(pose_frames, dtype=torch.float32)
    #     pose_frames_ = pose_frames_.permute(0, 3, 1, 2)  # 变换为 (T , C, H, W)
    #     pose_frames_ =  F.interpolate(pose_frames_, size=(H1, W1), mode='bilinear', align_corners=False)
    #     pose_frames_ = pose_frames_.view(T, C, H1, W1).permute(0, 2, 3, 1).numpy()  # 恢复为 (T, H1, W1, C)
    #     #crop
    #     top,down,left,right = int(min_H*random_value),  min(H,int(max_H*random_value)), int(min_W*random_value), min(W,int(max_W*random_value))
    #     cropped = pose_frames_[:, top:down, left:right, :]
    #     #translation
    #     _, crop_height, crop_width, _ = cropped.shape
    #     if (H-crop_height)>0 and (W-crop_width)>0:
    #         top_back = np.random.randint(0, (H-crop_height))
    #         left_back = np.random.randint(0, (W-crop_width))
    #         cropped_padded[:, top_back:top_back+crop_height, left_back:left_back+crop_width, :] = cropped
    #         # print("cropped_padded:",cropped_padded.shape)
    #         pose_frames = cropped_padded

    return pose_frames


def draw_rtmpose(
    canvas, points, scores, kpt_thr=0.4, stickwidth=2, r=2, pose_dict=None
):
    # 1. body
    mmpose_idx = [0, 17, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16, 1, 2, 3, 4]
    openpose_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    # 按照 openpose 的顺序重新排列 body_points 和 body_scores
    body_points_mmpose = np.zeros_like(points)
    body_scores_mmpose = np.zeros_like(scores)

    for i in range(len(mmpose_idx)):
        body_points_mmpose[mmpose_idx[i]] = points[openpose_idx[i]]
        body_scores_mmpose[mmpose_idx[i]] = scores[openpose_idx[i]]

    canvas = draw_body2(
        canvas,
        body_points_mmpose,
        body_scores_mmpose,
        kpt_thr=kpt_thr,
        stickwidth=stickwidth,
        r=r,
    )

    if pose_dict is not None:
        hands_left = pose_dict["hands_left"]
        hands_right = pose_dict["hands_right"]
        hands_left_score = pose_dict["hands_left_score"]
        hands_right_score = pose_dict["hands_right_score"]
        feet_left = pose_dict["feet_left"]
        feet_left_score = pose_dict["feet_left_score"]
        feet_right = pose_dict["feet_right"]
        feet_right_score = pose_dict["feet_right_score"]
        # # 2. hands
        stickwidth_hand = int(stickwidth * 0.75)
        canvas = draw_hand(
            canvas,
            hands_left,
            hands_left_score,
            kpt_thr=kpt_thr,
            stickwidth=stickwidth,
            r=r,
        )  # int(1.5*math.ceil(H/512))
        canvas = draw_hand(
            canvas,
            hands_right,
            hands_right_score,
            kpt_thr=kpt_thr,
            stickwidth=stickwidth,
            r=r,
        )

        # # 3. feet
        canvas = draw_foot(
            canvas,
            feet_left,
            feet_left_score,
            color=[255, 0, 85],
            kpt_thr=kpt_thr,
            stickwidth=stickwidth,
            r=r,
        )
        canvas = draw_foot(
            canvas,
            feet_right,
            feet_right_score,
            color=[255, 255, 0],
            kpt_thr=kpt_thr,
            stickwidth=stickwidth,
            r=r,
        )

    return canvas


def draw_body2(canvas, points, scores, kpt_thr=0.4, stickwidth=4, r=4):
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


def idx_to_rgb(idx):
    # color_list = ['#0d9da2','#2b87a7','#92e1cf','#b21646','#6e5743','#43a8d8','#c0c170','#26408a','#855e4a','#f70514','#6c4898','#c60716','#6951d9','#38c62a','#a10f9b','#709e33','#d3a1c3','#c6b306','#e2683a','#3e2ee3','#3d878b','#42724b','#94358b','#6b5c41','#8ff9c6','#b0d961','#20cf26','#3dfcfc','#e783ef','#83ae59','#4c3558','#3d4557','#ce15c2','#566430','#4365ad','#d8e891','#2f0f45','#2567dd','#82ddef','#666b06','#f954f3','#200dae','#2077eb','#10147c','#dbc264','#0d72de','#b91be0','#e12bce','#d8473a','#d08e78','#f870ee','#334b07','#3ff4f8','#e50d1f','#fc7606','#81c235','#8dc2ec','#289b04','#e5516a','#12e763','#82f4d0','#407d71','#869738','#95491a','#ca0251','#7282ed','#6cbe9a','#91b7fd','#f433e4','#340e25','#f153b5','#fff900','#076e05','#0f315f','#9f6292','#c670cf','#dc3a4d','#ac62bc','#9b362d','#1bb594','#e8fd4f','#324c58','#05c1c7','#19af62','#d8e27e','#b932ce','#0c84d5','#0dda0f','#8be866','#cf65d0','#391f55','#1f4ab7','#594864','#a41837','#c2a65d','#b4ba17','#4899df','#6be66a','#47e5e4','#d02abf']
    # color_list = '#3f031a', '#9c762e', '#f2b783', '#6d2dd7', '#104b1d', '#8427a6', '#861747', '#ef0f30', '#41976d', '#071ba7', '#980e88', '#648b19', '#6493a8', '#d20799', '#4a3588', '#91157a', '#1d4d00', '#f8dea4', '#3b7389', '#ef2338', '#c9da60', '#e70892', '#cb2a59', '#09db6f', '#3c8ce3', '#f01790', '#e9931b', '#1d963b', '#c91d5c', '#4ed6e8', '#10de45', '#a24972', '#ceb229', '#20bace', '#a26b89', '#e9bc10', '#b7df08', '#231964', '#fedc8f', '#f621d9', '#cc07e9', '#5d48c5', '#542dc3', '#6e2e63', '#9fca4f', '#306c66', '#1ef6b1', '#b90343', '#0e8293', '#a778ad', '#c5a81b', '#bbae1f', '#890796', '#b9a938', '#bc50b6', '#636eba', '#00e09a', '#096b8a', '#f0d1f8', '#57f772'
    # color_list = '#e9aac7', '#e1c93c', '#bd6e4a', '#2a23a5', '#f153dd', '#56166f', '#ac337e', '#fa57fe', '#0e5b62', '#df1de2', '#9334fa', '#004d3c', '#d1b831', '#c8ff63', '#ab99e7', '#0dcae9', '#06a64d', '#a03091', '#f057d4', '#8415ee', '#6c67e0', '#b2ea32', '#91852e', '#808dc4', '#0f96e9', '#05c5ff', '#25f358', '#a68826', '#7ee741', '#eb6244', '#e7851a', '#e655ea', '#5adb6f', '#0eaa03', '#658ed3', '#f3105c', '#82ee0c', '#f9596e', '#5a3585', '#7bb045', '#547a77', '#514632', '#9f284f', '#8b861a', '#5785b0', '#d461d1', '#bb7834', '#73b06d', '#02561a', '#84fbca', '#7a6d9a', '#1bd4c5', '#2bb1b0', '#0a5575', '#8961ca', '#fd0987', '#40cc91', '#734950', '#59eafd', '#8e9b6a'
    color_list = (
        "#e9aac7",
        "#e1c93c",
        "#bd6e4a",
        "#2a23a5",
        "#f153dd",
        "#56166f",
        "#ac337e",
        "#ef0f30",
        "#41976d",
        "#071ba7",
        "#980e88",
        "#648b19",
        "#6493a8",
        "#d20799",
        "#4a3588",
        "#91157a",
        "#1d4d00",
        "#f8dea4",
        "#3b7389",
        "#ef2338",
        "#c9da60",
        "#e70892",
        "#cb2a59",
        "#09db6f",
        "#3c8ce3",
        "#f01790",
        "#e9931b",
        "#1d963b",
        "#c91d5c",
        "#4ed6e8",
        "#10de45",
        "#a24972",
        "#ceb229",
        "#20bace",
        "#a26b89",
        "#e9bc10",
        "#b7df08",
        "#231964",
        "#fedc8f",
        "#f621d9",
        "#cc07e9",
        "#5d48c5",
        "#542dc3",
        "#6e2e63",
        "#9fca4f",
        "#306c66",
        "#1ef6b1",
        "#b90343",
        "#0e8293",
        "#a778ad",
        "#c5a81b",
        "#bbae1f",
        "#890796",
        "#b9a938",
        "#bc50b6",
        "#636eba",
        "#00e09a",
        "#096b8a",
        "#f0d1f8",
        "#57f772",
    )

    return color_list[idx]


def get_color_dict(joint):
    key_to_color = {}
    i = 0
    for key, value in joint.items():
        for vv in value:
            # key_to_color[str(key)+"_"+str(vv)] = string_to_rgb(str(key)+"_"+str(vv))
            key_to_color[str(key) + "_" + str(vv)] = idx_to_rgb(i)
            i += 1
    return key_to_color


def draw_3dpose(pose3d, key_to_color, width, height, elev=None, azim=None):
    fig = plt.figure(dpi=100, figsize=(width / 100, height / 100))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.view_init(elev=elev, azim=azim)  # elev: 仰角, azim: 方位角
    # 禁用网格
    ax.grid(False)
    # 控制坐标轴范围
    ax.set_xlim(-0.84, 0.84)  # 只显示x在(-0.84, 0.84)之间的数据
    ax.set_ylim(-0.84, 0.84)  # 只显示y在(-0.84, 0.84)之间的数据
    ax.set_zlim(0, 1)  # 只显示y在(-0.84, 0.84)之间的数据

    if "bodies" in pose3d:
        bodies = pose3d["bodies"]
        sc = ax.scatter(
            bodies[:, 0], bodies[:, 1], bodies[:, 2], s=2, color="white", linewidths=0
        )
        per_nums = bodies.shape[0] // 18
        for per_num in range(per_nums):
            skeleton = bodies[per_num * 18 : per_num * 18 + 18]
            for key, value in joint.items():
                if key >= 18:
                    continue
                for vv in value:
                    if -1 in [
                        skeleton[key, 0],
                        skeleton[vv, 0],
                        skeleton[key, 1],
                        skeleton[vv, 1],
                        skeleton[key, 2],
                        skeleton[vv, 2],
                    ]:
                        continue
                    ax.plot(
                        [skeleton[key, 0], skeleton[vv, 0]],
                        [skeleton[key, 1], skeleton[vv, 1]],
                        [skeleton[key, 2], skeleton[vv, 2]],
                        color=key_to_color[str(key) + "_" + str(vv)],
                        label="3D line",
                    )

    if "hands" in pose3d:
        hands = pose3d["hands"]
        sc = ax.scatter(
            hands[:, 0], hands[:, 1], hands[:, 2], s=2, color="white", linewidths=0
        )
        per_nums = hands.shape[0] // 42
        for per_num in range(per_nums):
            skeleton = hands[per_num * 42 : per_num * 42 + 42]
            for key, value in joint.items():
                if key < 18:
                    continue
                for vv in value:
                    if -1 in [
                        skeleton[key - 18, 0],
                        skeleton[vv - 18, 0],
                        skeleton[key - 18, 1],
                        skeleton[vv - 18, 1],
                        skeleton[key - 18, 2],
                        skeleton[vv - 18, 2],
                    ]:
                        continue
                    ax.plot(
                        [skeleton[key - 18, 0], skeleton[vv - 18, 0]],
                        [skeleton[key - 18, 1], skeleton[vv - 18, 1]],
                        [skeleton[key - 18, 2], skeleton[vv - 18, 2]],
                        color=key_to_color[str(key) + "_" + str(vv)],
                        label="3D line",
                    )

    if "faces" in pose3d:
        faces = pose3d["faces"]
        sc = ax.scatter(
            faces[:, 0], faces[:, 1], faces[:, 2], s=2, color="white", linewidths=0
        )

    # 获取图像数据
    fig.canvas.get_renderer()
    image_data = fig.canvas.tostring_rgb()

    # 转换为array
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image_array = image_array.reshape((1080, 1920, 3))

    img = Image.fromarray(image_array.astype("uint8"))
    # 如果需要，保存图像
    # pil_image.save('output.png')
    plt.close()
    return img


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
        # if i == 15 or i == 16:
        #     if score < 0.6:
        #         continue
        # if score < kpt_thr:
        #     continue
        x, y = points[i][0:2]
        x, y = int(x), int(y)
        cv2.circle(canvas, (x, y), r, colors[i], thickness=-1)
        # cv2.putText(canvas, i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    # draw line
    for i in range(len(connetions)):
        i = 16 - i
        point1_idx, point2_idx = connetions[i][0:2]

        # if point2_idx == 15 or point2_idx == 16:
        #     if scores[point2_idx] < 0.6:
        #         continue

        # if scores[point1_idx] < kpt_thr or scores[point2_idx] < kpt_thr:
        #     continue

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


def draw_body_(canvas, points, scores, kpt_thr=0.4, stickwidth=2, r=2):
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


def draw_foot(
    canvas, foot, scores, color=[255, 0, 255], kpt_thr=0.2, stickwidth=1, r=1
):
    if any(score < kpt_thr for score in scores):
        return canvas
    x, y = foot[2][0:2]
    x1, y1 = foot[0][0:2]
    x2, y2 = foot[1][0:2]
    x, y = int(x), int(y)
    mx = int((x1 + x2) / 2)
    my = int((y1 + y2) / 2)

    cv2.circle(canvas, (x, y), r, [255, 0, 0], thickness=-1)
    cv2.circle(canvas, (mx, my), r, [255, 0, 0], thickness=-1)
    cv2.line(canvas, (x, y), (mx, my), color, stickwidth, lineType=cv2.LINE_AA)
    return canvas


def draw_hand(canvas, hand, scores, kpt_thr=0.4, stickwidth=1, r=1):
    # 16 点
    color_finger = [
        [255, 0, 0],  # 大拇指颜色
        [0, 255, 0],  # 食指颜色
        [255, 0, 255],  # 中指颜色
        [0, 255, 255],  # 无名指颜色
        [255, 255, 0],  # 小拇指颜色
    ]

    # cv2.putText(canvas, i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0], 1)

    # 绘制连接线
    for i in range(5):
        if scores[1 + i * 3] >= kpt_thr and scores[0] >= kpt_thr:
            cv2.line(
                canvas,
                tuple(map(int, hand[0][0:2])),
                tuple(map(int, hand[1 + i * 3][0:2])),
                color_finger[i],
                stickwidth,
                lineType=cv2.LINE_AA,
            )

        for j in range(1, 3):
            if (
                scores[1 + i * 3 + j - 1] >= kpt_thr
                and scores[1 + i * 3 + j] >= kpt_thr
            ):
                start_point = hand[1 + i * 3 + j - 1][0:2]
                end_point = hand[1 + i * 3 + j][0:2]
                cv2.line(
                    canvas,
                    tuple(map(int, start_point)),
                    tuple(map(int, end_point)),
                    color_finger[i],
                    stickwidth,
                    lineType=cv2.LINE_AA,
                )

        # 绘制手部关键点
    for i in range(len(hand)):
        score = scores[i]
        if score < kpt_thr:
            continue
        x, y = hand[i][0:2]
        x, y = int(x), int(y)
        cv2.circle(canvas, (x, y), r, [0, 0, 255], thickness=-1)

    return canvas


def check_hand(hand):
    finger_indices = [
        [0, 1, 2, 3],  # 大拇指
        [0, 4, 5, 6],  # 食指
        [0, 7, 8, 9],  # 中指
        [0, 10, 11, 12],  # 无名指
        [0, 13, 14, 15],  # 小拇指
    ]
    finger_lengths = []

    for finger in finger_indices:
        finger_length = 0
        for i in range(len(finger) - 1):
            point_a = np.array(hand[finger[i]][0:2])
            point_b = np.array(hand[finger[i + 1]][0:2])
            distance = np.linalg.norm(point_a - point_b)
            finger_length += distance
        finger_lengths.append(finger_length)

    finger_thresholds = [1, 0.65, 0.6, 0.65, 1]

    for i, length in enumerate(finger_lengths):
        other_lengths = finger_lengths[:i] + finger_lengths[i + 1 :]
        mean_length = np.mean(other_lengths)
        threshold_ratio = finger_thresholds[i]

        if length > mean_length * (1 + threshold_ratio) or length < mean_length * (
            1 - threshold_ratio
        ):
            return False
    return True
