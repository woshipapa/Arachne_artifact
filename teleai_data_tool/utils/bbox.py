from typing import List, Tuple, Union

import cv2
import numpy as np


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (left, top, right, bottom)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """
    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_xywh2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
            as (x, y, h, w)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
            (n, 2)
        - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
            (n, 2)
    """

    # convert single bbox from (4, ) to (1, 4)
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x, y, w, h = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x + w * 0.5, y + h * 0.5])
    scale = np.hstack([w, h]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def bbox_cs2xyxy(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x,y,w,h).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x, y, w, h) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, xy + wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_cs2xywh(
    center: np.ndarray, scale: np.ndarray, padding: float = 1.0
) -> np.ndarray:
    """Transform the bbox format from (center, scale) to (x,y,w,h).

    Args:
        center (ndarray): BBox center (x, y) in shape (2,) or (n, 2)
        scale (ndarray): BBox scale (w, h) in shape (2,) or (n, 2)
        padding (float): BBox padding factor that will be multilied to scale.
            Default: 1.0

    Returns:
        ndarray[float32]: BBox (x, y, w, h) in shape (4, ) or (n, 4)
    """

    dim = center.ndim
    assert scale.ndim == dim

    if dim == 1:
        center = center[None, :]
        scale = scale[None, :]

    wh = scale / padding
    xy = center - 0.5 * wh
    bbox = np.hstack((xy, wh))

    if dim == 1:
        bbox = bbox[0]

    return bbox


def bbox_xyxy2xywh(bbox_xyxy: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
    """convert bbox from xyxy to xywh.

    Args:
        bbox_xyxy (Union[List, np.ndarray]): bbox in format list [x, y, x, y]
        or numpy array with shape [n, 4]

    Returns:
        Union[List, np.ndarray]: bbox in format list [x, y, w, h]
        or numpy array with shape [n, 4]
    """
    bbox_xywh = bbox_xyxy.copy()
    if isinstance(bbox_xyxy, list):
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0]
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1]
    else:
        bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
        bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh: Union[List, np.ndarray]) -> Union[List, np.ndarray]:
    """convert bbox from xywh to xyxy.

    Args:
        bbox_xywh (Union[List, np.ndarray]): bbox in format list [x, y, w, h]
        or numpy array with shape [n, 4]

    Returns:
        Union[List, np.ndarray]: bbox in format list [x, y, x, y]
        or numpy array with shape [n, 4]
    """
    bbox_xyxy = bbox_xywh.copy()
    if isinstance(bbox_xyxy, list):
        bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0]
        bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1]
    else:
        bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
        bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]
    return bbox_xyxy


def kpt_to_bbox(
    kpt: np.ndarray, scale_x: float = 1.5, scale_y: float = 1.4
) -> np.ndarray:
    """get bounding box from keypoints.

    Args:
        kpt (np.ndarray): keypoint with shape [n, 2]
        scale_x (float, optional): expand scale for bbox width.
        Defaults to 1.5.
        scale_y (float, optional): expand scale for bbox height.
        Defaults to 1.4.

    Returns:
        np.ndarray: bounding bbox in format [x, y, w, h]
    """
    min_x = kpt[:, 0].min()
    min_y = kpt[:, 1].min()
    max_x = kpt[:, 0].max()
    max_y = kpt[:, 1].max()
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    min_x = (min_x - cx) * scale_x + cx
    min_y = (min_y - cy) * scale_y + cy
    max_x = (max_x - cx) * scale_x + cx
    max_y = (max_y - cy) * scale_y + cy
    bbox = np.array([min_x, min_y, max_x - min_x, max_y - min_y], dtype=np.float32)
    return bbox


def get_IoU(pred_bbox, gt_bbox):
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    inters = iw * ih

    uni = (
        (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0)
        + (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0)
        - inters
    )

    overlaps = inters / uni

    return overlaps


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c
