import os
import cv2
import numpy as np
import scipy
import hashlib
from scipy.ndimage import zoom


def resize_matrix(original_matrix, height, width):
    # x = np.arange(original_matrix.shape[0])
    # y = np.arange(original_matrix.shape[1])

    # x_new = np.linspace(0, original_matrix.shape[0] - 1, height)
    # y_new = np.linspace(0, original_matrix.shape[1] - 1, width)

    # xx, yy = np.meshgrid(x, y)
    # resized_matrix = interpn((xx.ravel(), yy.ravel()), original_matrix.ravel(),(x_new, y_new), method='nearest')
    # resized_matrix = resized_matrix.reshape(height, width)
    # return resized_matrix
    zoom_factors = (height / original_matrix.shape[0], width / original_matrix.shape[1])
    resized_matrix = zoom(original_matrix, zoom_factors)
    return resized_matrix


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
            try:
                if (
                    idx_img < len(segment[key])
                    and segment[key][idx_img] != []
                    and os.path.exists(segment[key][idx_img])
                ):
                    dense_matrix = scipy.sparse.load_npz(
                        segment[key][idx_img]
                    ).toarray()
                    if np.sum(dense_matrix) < (height * width / 150.0):
                        continue
                    # dense_matrix = resize_matrix(dense_matrix, height, width)
                    if (
                        dense_matrix.shape[0] != height
                        or dense_matrix.shape[1] != width
                    ):
                        dense_matrix = resize_matrix(dense_matrix, height, width)
                    img[dense_matrix, :] += np.array(
                        string_to_rgb(key.split("#")[0]), dtype=np.uint8
                    )
            except Exception:
                continue
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


def string_to_rgb(color_name):
    hash_object = hashlib.md5(color_name.encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    r = r % 256
    g = g % 256
    b = b % 256
    return (r, g, b)
