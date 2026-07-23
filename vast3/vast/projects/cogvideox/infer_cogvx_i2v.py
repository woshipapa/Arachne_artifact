import math
import os
import sys
from itertools import accumulate

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from tqdm import tqdm


from vast.pipelines import CogVideoXPipeline
from vast.pipelines.vision.mask import get_box_frames, get_segment_frames


def load_video(
    video,
    valid_range=None,
    sample_frames=None,
    sample_stride=1,
    sample_method=2,
    max_frames=None,
):
    if sample_frames is not None:
        assert max_frames is None
    if valid_range is None:
        valid_range = (0, len(video))
    video_length = valid_range[1] - valid_range[0]
    if sample_frames is None:
        sample_indexes = np.arange(
            valid_range[0], valid_range[1], sample_stride, dtype=int
        )
        if max_frames is not None and len(sample_indexes) > max_frames:
            sample_indexes = sample_indexes[:max_frames]
    elif sample_frames >= video_length:
        sample_indexes = np.arange(valid_range[0], valid_range[1], dtype=int)
    else:
        sample_length = min(video_length, (sample_frames - 1) * sample_stride + 1)
        sample_indexes = np.linspace(
            valid_range[0], valid_range[0] + sample_length - 1, sample_frames, dtype=int
        )
    images = sample_video(video, sample_indexes, sample_method)
    images = [Image.fromarray(image) for image in images]
    return images, sample_indexes


def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = (
            frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        )
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = (
            frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        )
        frames = frames[indexes]
    else:
        assert False
    return frames


def concat_images(images, direction="horizontal", pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == "horizontal":
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full(
            (height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype
        )
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == "vertical":
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full(
            (height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype
        )
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def concat_images_grid(images, cols, pad=0, pad_value=0):
    new_images = []
    while len(images) > 0:
        new_image = concat_images(images[:cols], pad=pad, pad_value=pad_value)
        new_images.append(new_image)
        images = images[cols:]
    new_image = concat_images(
        new_images, direction="vertical", pad=pad, pad_value=pad_value
    )
    return new_image


def get_image_size(src_size, dst_size, mode, max_size=None, multiple=None):
    width, height = src_size
    if mode in ("long", "short", "height", "width"):
        if isinstance(dst_size, (list, tuple)):
            assert dst_size[0] == dst_size[1]
            dst_size = dst_size[0]
        if mode == "long":
            scale = float(dst_size) / max(height, width)
        elif mode == "short":
            scale = float(dst_size) / min(height, width)
        elif mode == "height":
            scale = float(dst_size) / height
        elif mode == "width":
            scale = float(dst_size) / width
        dst_height = int(round(height * scale))
        dst_width = int(round(width * scale))
    elif mode in ("fixed", "outer_fit", "inner_fit"):
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        if mode == "outer_fit":
            if float(dst_height) / height > float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
        elif mode == "inner_fit":
            if float(dst_height) / height < float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
    elif mode == "area":
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        aspect_ratio = float(height) / float(width)
        dst_area = dst_height * dst_width
        dst_height = int(round(np.sqrt(dst_area * aspect_ratio)))
        dst_width = int(round(np.sqrt(dst_area / aspect_ratio)))
    else:
        assert False
    if max_size is not None and max(dst_height, dst_width) > max_size:
        if dst_height > dst_width:
            dst_width = int(round(float(max_size) / dst_height * dst_width))
            dst_height = max_size
        else:
            dst_height = int(round(float(max_size) / dst_width * dst_height))
            dst_width = max_size
    if multiple is not None:
        dst_height = int(round(dst_height / multiple)) * multiple
        dst_width = int(round(dst_width / multiple)) * multiple
    return dst_width, dst_height


def process_data_list(data_list, world_size=1, rank=0, indexes=None):
    if indexes is None:
        data_size = len(data_list)
        local_size = math.floor(data_size / world_size)
        local_size_list = [local_size for _ in range(world_size)]
        for i in range(data_size - local_size * world_size):
            local_size_list[i] += 1
        assert sum(local_size_list) == data_size
        local_size_list = [0] + list(accumulate(local_size_list))
        begin = local_size_list[rank]
        end = local_size_list[rank + 1]
        indexes = [i for i in range(begin, end)]
    print(f"Rank {rank} Data {len(indexes)}/{len(data_list)}: {indexes}")
    data_list = [data_list[i] for i in indexes]
    return data_list


def inference(device, world_size=1, rank=0):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # dst_size = (1280, 736)
    # num_frames = 97
    dst_size = (720, 480)
    num_frames = 49
    save_dir = os.path.join(cur_dir, './results_sjl/')
    # load model
    pipe = CogVideoXPipeline.from_pretrained(
        os.path.join(cur_dir, "../../../pretrained/models--THUDM--CogVideoX-5b"),
        # transformer_model_path=os.path.join(cur_dir, '../../../pretrained/cxz_720p_v1'),
        transformer_model_path=os.path.join(
            cur_dir,
            "/root/code/teleAI-T2V/vast/scripts/examples/cogvideox/experiments_sjl/cogvideox/cogvideox_20241216/models/checkpoint_epoch_2_step_30000/transformer_ema",
        ),
        torch_dtype=torch.bfloat16,
        scheduler_type="dpm",
    )
    pipe.to(device)
    # pose_pipe = gm_load_pipeline('keypoints/dwpose/yolox_l_coco_body_hand_face', lazy=True).to(local_device)
    # load data
    prompts = [
        "A small boy is having breakfast, putting a piece of fried egg into his mouth and chewing the food.",
        # "",
    ]
    data_list = []
    for i in range(len(prompts)):
        data_dict = dict(
            prompt=prompts[i],
            save_name="breakfast720p-3",
            save_fps=15,
            ref_image={
                0: Image.open(os.path.join(cur_dir, "../../../assets/breakfast2.jpg")),
                # 25: Image.open("assets/gra/1.png"),
                # 30: Image.open("assets/relation/1.png"),
                # 35: Image.open("assets/relation/1.png"),
                # -1: Image.open("assets/breakfast/10.jpg"),
            },
        )
        data_list.append(data_dict)
    data_list = process_data_list(data_list, world_size, rank)
    # inference
    dst_width, dst_height = dst_size
    zero_image = Image.new("RGB", (dst_width, dst_height), (0, 0, 0))
    ref_images = [zero_image for _ in range(num_frames)]
    for n in tqdm(range(len(data_list))):
        data_dict = data_list[n]
        prompt = data_dict["prompt"]

        if "ref_image" in data_dict:
            for ref_idx, ref_image in data_dict["ref_image"].items():
                height, width = ref_image.height, ref_image.width
                dst_width, dst_height = dst_size
                # dst_width, dst_height = get_image_size(
                #     (width, height), dst_size, mode='area', multiple=16
                # )
                ref_image = F.resize(
                    ref_image, (dst_height, dst_width), InterpolationMode.BILINEAR
                )
                ref_images[ref_idx] = ref_image

        # inference
        print(prompt, dst_height, dst_width)
        output_images = pipe(
            prompt=prompt,
            height=dst_height,
            width=dst_width,
            num_frames=num_frames,
            num_inference_steps=25,
            seed=333,
            ref_images=ref_images,
            # use_dynamic_cfg=True,
        ).frames[0]
        # save results
        save_name = data_dict["save_name"]
        save_fps = data_dict["save_fps"]
        vis_images = []
        for k in range(len(output_images)):
            if ref_images is None:
                vis_image = [output_images[k]]
            else:
                vis_image = [ref_images[0], output_images[k]]
            vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
            # vis_images.append(vis_image)
            vis_images.append(output_images[k])
        save_path = os.path.join(save_dir, "{}.mp4".format(save_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, vis_images, fps=save_fps)


def infer_cogvx_i2v():
    inference(device="cuda")

    # multiprocessing.set_start_method('spawn')
    # process_list = []
    # gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # world_size = len(gpu_ids)
    # for i in range(world_size):
    #     device = f'cuda:{gpu_ids[i]}'
    #     rank = i
    #     process = Process(target=inference, args=(device, world_size, rank))
    #     process.start()
    #     process_list.append(process)
    # for process in process_list:
    #     process.join()


if __name__ == "__main__":
    infer_cogvx_i2v()
