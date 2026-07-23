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

def init_paths(project_name=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    python_paths = [
        '../../'
    ]
    if project_name is not None:
        python_paths.append(os.path.join(cur_dir, project_name))
    for python_path in python_paths:
        sys.path.insert(0, python_path)
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += ':{}'.format(python_path)
        else:
            os.environ['PYTHONPATH'] = python_path
init_paths()

from vast.pipelines import VASTBASEPipeline

def concat_images(images, direction='horizontal', pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == 'horizontal':
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == 'vertical':
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
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
    new_image = concat_images(new_images, direction='vertical', pad=pad, pad_value=pad_value)
    return new_image

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
    print(f'Rank {rank} Data {len(indexes)}/{len(data_list)}: {indexes}')
    data_list = [data_list[i] for i in indexes]
    return data_list


def inference(device, world_size=1, rank=0):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dst_size = (1280, 736)
    num_frames = 49
    # dst_size = (736, 480)
    # num_frames = 49
    save_dir = os.path.join(cur_dir, './results_vast/')
    # load model
    pipe = VASTBASEPipeline.from_pretrained(
        os.path.join(cur_dir, '../../pretrained/models--THUDM--CogVideoX-5b'),
        # transformer_model_path=os.path.join(cur_dir, '../../../pretrained/cxz_720p_v2'),
        transformer_model_path=os.path.join(cur_dir, '/root/code/vast/projects/vastbase/experiments_sjl/vast/vast_20250108_ftcog/models/checkpoint_epoch_16_step_250/transformer_ema'),
        torch_dtype=torch.bfloat16,
        scheduler_type='dpm',
        # from_giga=True,
        patch_size=[(1,2,2),(1,4,4)],
        only_orgin=True
    )
    pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    # load data
    prompts = [
        "A small boy is having breakfast, putting a piece of fried egg into his mouth and chewing the food.",
        # "",
    ]
    data_list = []
    for i in range(len(prompts)):
        data_dict = dict(
            prompt=prompts[i],
            save_name='breakfast736p-ft-rebase',
            save_fps=15,
            ref_image={
                0: Image.open(os.path.join(cur_dir, "../../test/assets/breakfast2.jpg")), 
                # 25: Image.open("assets/gra/1.png"), 
                # 30: Image.open("assets/relation/1.png"), 
                # 35: Image.open("assets/relation/1.png"), 
                # -1: Image.open("assets/breakfast/10.jpg"), 
            }
        )
        data_list.append(data_dict)
    data_list = process_data_list(data_list, world_size, rank)
    # inference
    dst_width, dst_height = dst_size
    zero_image = Image.new('RGB', (dst_width, dst_height), (0, 0, 0))
    ref_images = [zero_image for _ in range(num_frames)]
    for n in tqdm(range(len(data_list))):
        data_dict = data_list[n]
        prompt = data_dict['prompt']

        if 'ref_image' in data_dict:
            for ref_idx, ref_image in data_dict['ref_image'].items():
                height, width = ref_image.height, ref_image.width
                dst_width, dst_height = dst_size
                ref_image = F.resize(ref_image, (dst_height, dst_width), InterpolationMode.BILINEAR)
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
        save_name = data_dict['save_name']
        save_fps = data_dict['save_fps']
        vis_images = []
        for k in range(len(output_images)):
            if ref_images is None:
                vis_image = [output_images[k]]
            else:
                vis_image = [ref_images[0], output_images[k]]
            vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
            # vis_images.append(vis_image)
            vis_images.append(output_images[k])
        save_path = os.path.join(save_dir, '{}.mp4'.format(save_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, vis_images, fps=save_fps)


def infer_vast_i2v():
    inference(device='cuda')

if __name__ == '__main__':
    infer_vast_i2v()
