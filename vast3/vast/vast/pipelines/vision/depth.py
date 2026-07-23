# ref: https://github.com/DepthAnything/Depth-Anything-V2
"""
usage:
export HF_HOME=/root/share/
"""

from transformers import pipeline
from PIL import Image
import os

os.system("export HF_HOME=/root/share/")


def get_pipe():
    return pipeline(
        task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf"
    )


def convert_to_pointcloud():
    # Load the image
    color_image = Image.open(filename).convert("RGB")
    width, height = color_image.size

    # Read the image using OpenCV
    image = cv2.imread(filename)
    pred = depth_anything.infer_image(image, height)

    # Resize depth prediction to match the original image size
    resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / args.focal_length_x
    y = (y - height / 2) / args.focal_length_y
    z = np.array(resized_pred)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(color_image).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def pose_3D(
    points2d, depth_map, focal_length_x=1000, focal_length_y=1000, normalize=False
):
    """
    @param:
    points2d: numpy array of shape (n,2), refers to h,w
    depth_map: numpy array of shape (h,w)
    K: intrinsic parameter of camera, will create a default value if set to None
    normalize: whether constriant 3D points with human prior
    """
    height, width = depth_map.shape[0], depth_map.shape[1]
    points2d[:0] = (points2d[:0] - height / 2) / focal_length_x
    points2d[:1] = (points2d[:1] - width / 2) / focal_length_y
    import pdb

    pdb.set_trace()
    z = depth_map
    points3d = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(
        -1, 3
    )


if __name__ == "__main__":
    pipe = get_pipe()
    img = Image.open()
