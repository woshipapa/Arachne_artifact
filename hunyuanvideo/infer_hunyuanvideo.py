import json
import re
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import partial

from diffusers.utils import export_to_video
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
from tqdm import tqdm
from vast.pipelines import HunyuanVideoPipeline


# Constants
MODEL_TYPE_MAPPING = {
    16: "t2v",
    32: "i2v",
    33: "i2vhy"
}
DEFAULT_CONFIG = {
    "dst_size": (1280, 720),
    "guidance_scale": 6.0,
    "embedded_guidance_scale": 1.0,
    "save_fps": 15,
    "num_frames": 49,
    "inference_steps": 50,
    "base_model": "<CKPT_ROOT>/hunyuanvideo_13b"
}


class InferenceConfig:
    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        dst_size: Tuple[int, int] = DEFAULT_CONFIG["dst_size"],
        guidance_scale: float = DEFAULT_CONFIG["guidance_scale"],
        embedded_guidance_scale: float = DEFAULT_CONFIG["embedded_guidance_scale"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        inference_steps: int = DEFAULT_CONFIG["inference_steps"],
        base_model: str = DEFAULT_CONFIG["base_model"]
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.dst_size = dst_size
        self.guidance_scale = guidance_scale
        self.embedded_guidance_scale = embedded_guidance_scale
        self.save_fps = save_fps
        self.num_frames = num_frames
        self.inference_steps = inference_steps
        self.base_model = base_model


def get_parent_path(path: str, level: int = 3) -> Path:
    return Path(path).resolve().parents[level-1]


def get_checkpoints(model_dir: str, interval: int = 1) -> List[Tuple[str, str]]:
    checkpoints = []
    pattern = re.compile(r"checkpoint_epoch_(\d+)_step_(\d+)")
    models_dir = Path(model_dir) / "models"
    
    for dir_path in models_dir.glob("**/checkpoint_*"):
        if match := pattern.search(dir_path.name):
            epoch, step = match.groups()
            if int(step) % interval == 0:
                name = f"{Path(model_dir).name}_ep{epoch}iter{step}"
                trans_path = dir_path / "transformer"
                if trans_path.exists():
                    checkpoints.append((name, str(trans_path)))
    return checkpoints


def generate_video_filename(result_name: str, model_name: str, config: InferenceConfig) -> str:
    width, height = config.dst_size
    base_name = f"{result_name}_{model_name}_{height}p_f{config.num_frames}_s{config.inference_steps}_fps{config.save_fps}_g{config.guidance_scale}_eg{config.embedded_guidance_scale}"
    return f"{base_name}.mp4"


def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    width, height = config.dst_size
    ref_images = [Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(config.num_frames)]
    
    for frame_idx, img_path in config.ref_images.items():
        if frame_idx >= config.num_frames:
            raise ValueError(f"Reference frame index {frame_idx} exceeds the frame count {config.num_frames}")
        
        original_img = Image.open(img_path)
        resized_img = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)
        ref_images[frame_idx] = resized_img
    
    return ref_images


def load_pipeline(model_path: str, config: InferenceConfig, device: torch.device) -> HunyuanVideoPipeline:
    config_path = get_parent_path(model_path) / "config.json"
    with open(config_path) as f:
        in_channels = json.load(f)["models"]["transformer"]["in_channels"]
    
    model_type = MODEL_TYPE_MAPPING.get(in_channels)
    if not model_type:
        raise ValueError(f"Unsupported in_channels: {in_channels}")
    
    pipe = HunyuanVideoPipeline.from_pretrained(
        config.base_model,
        transformer_model_path=model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.vae.enable_tiling()
    return pipe


def inference_worker(
    rank: int,
    world_size: int,
    model_list: List[Tuple[str, str]],
    config: InferenceConfig,
    gpu_ids: List[int],
    result_name: str
):
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    save_dir = Path(__file__).parent / "results" / result_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_process = model_list[rank::world_size]
    if not models_to_process:
        print(f"[Rank {rank}] No models to process")
        return
    
    ref_images = prepare_reference_images(config)
    
    for model_name, model_path in models_to_process:
        try:
            save_name = generate_video_filename(result_name, model_name, config)
            save_path = save_dir / save_name
            
            if save_path.exists():
                print(f"[Rank {rank}] skipping existing file: {save_path}")
                continue
                
            pipe = load_pipeline(model_path, config, device)
            
            output = pipe(
                prompt=config.prompt,
                height=config.dst_size[1],
                width=config.dst_size[0],
                num_frames=config.num_frames,
                num_inference_steps=config.inference_steps,
                seed=42,
                model_type=MODEL_TYPE_MAPPING[pipe.transformer.config.in_channels],
                ref_images=ref_images,
                guidance_scale=config.guidance_scale,
                embedded_guidance_scale=config.embedded_guidance_scale,
            ).frames[0]
            
            export_to_video(output, str(save_path), fps=config.save_fps)
            print(f"[Rank {rank}] saved: {save_path}")
            
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Rank {rank}] error while processing model {model_name}: {str(e)}")


def run_inference_pipeline(
    checkpoint_dirs: List[str],
    prompt_configs: Dict[str, Dict[str, str]],
    gpu_ids: List[int] = None,
    checkpoint_interval: int = 100
):
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)
    
    all_models = []
    for dir_path in checkpoint_dirs:
        all_models.extend(get_checkpoints(dir_path, checkpoint_interval))
    
    if not all_models:
        raise ValueError("No valid checkpoint found")
    
    for test_name, config in prompt_configs.items():
        print(f"\n{'='*40}\nProcessing test case: {test_name}\n{'='*40}")
        result_name = Path(config["ref_images"][0]).stem
        
        inference_config = InferenceConfig(
            prompt=config["prompt"],
            ref_images=config["ref_images"],
            **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"}
        )
        
        mp.spawn(
            partial(inference_worker,
                   world_size=world_size,
                   model_list=all_models,
                   config=inference_config,
                   gpu_ids=gpu_ids,
                   result_name=result_name),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    CHECKPOINT_DIRS = [
        "<CKPT_ROOT>/i2vhy_480p_81_dp320",
    ]

    PROMPT_CONFIGS = {
        "oven":{
            "prompt": """A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door.""",
            "ref_images": {0: "<CKPT_ROOT>/oven.jpg"},
        },
        "kids2": {
            "prompt":  """Two young boys, one with blonde hair in a yellow shirt and blue jeans, the other with light brown hair in a red shirt and dark shorts, are seated on a grassy field, sipping water from clear plastic bottles. They are surrounded by the tranquility of nature, with a yellow bicycle and its black frame lying on its side nearby, suggesting a recent ride. The boys appear relaxed and engaged in conversation, enjoying a peaceful moment of rest in the late afternoon sun.""",
            "ref_images": {0: "<CKPT_ROOT>/kids2.jpg"},
        },
        "talk": {
            "prompt":  """A woman with blonde hair styled in a high bun, wearing a white sweater and black pants, is seated on a wooden chair in front of a microphone on a stage. She is engaged in a conversation with a man in a black suit, who is seated opposite her. The stage is illuminated by warm, ambient lighting, and the background features a cityscape with tall buildings and colorful lights. The woman appears to be speaking and gesturing animatedly, while the man listens and occasionally responds. The main subjects are the woman and the man. The woman has blonde hair styled in a high bun, wears a white sweater, and black pants. She is seated on a wooden chair and is positioned in front of a microphone. The man, dressed in a black suit, is seated opposite her, facing her. He is holding a notepad and pen, and he occasionally gestures with his hands while speaking. The woman is actively speaking and gesturing, moving her hands and head as she talks. The man occasionally responds, nodding and gesturing with his hands. The background remains static, with no visible movement. The camera is stationary, capturing a medium shot of the two subjects from a frontal view.""",
            "ref_images": {0: "<CKPT_ROOT>/talk.jpg"},
        },
        "work": {
            "prompt":  """Three professionals, two men in suits and a woman in a black blazer, are engaged in a serious discussion at an outdoor construction site. They are focused on a clipboard and documents, suggesting a meeting about a construction project. The setting includes towering cranes and an overcast sky, indicating a professional atmosphere. Throughout the video, the woman points to specific details on the clipboard, actively participating in the conversation. The men, one in a dark suit and the other in a navy blue suit, listen attentively, reflecting a collaborative and professional exchange of ideas.""",
            "ref_images": {0: "<CKPT_ROOT>/work.jpg"},
        },
        "kids": {
            "prompt":  """Two young boys, one with blonde hair in a yellow shirt and blue jeans, the other with light brown hair in a red shirt and dark shorts, are seated on a grassy field, sipping water from clear plastic bottles. They are surrounded by the tranquility of nature, with a yellow bicycle and its black frame lying on its side nearby, suggesting a recent ride. The boys appear relaxed and engaged in conversation, enjoying a peaceful moment of rest in the late afternoon sun.""",
            "ref_images": {0: "<CKPT_ROOT>/kids.jpg"},
        },
        "naobaijin_cn": {
            "prompt":  """Two cartoon characters dance on the spot. On the left is a man in a hat with a moustache, wearing traditional patterned clothing; on the right is a woman in a headscarf with earrings and other jewellery, wearing folk-style dress. Both are dancing, and the man falls over as he dances.""",
            "ref_images": {0: "<CKPT_ROOT>/naobaijin.jpg"},
        },
        "hula": {
            "prompt":  """the woman is hula hooping outdoors.""",
            "ref_images": {0: "<CKPT_ROOT>/hula.jpg"},
        },
        "hamburger":{
            "prompt":  """A woman is dining in a fast-food restaurant. She is wearing transparent disposable gloves and holding a hamburger with both hands, bringing it to her mouth. There are other foods placed on the dining table in front of her. In the background, the ordering screens of the fast - food restaurant can be seen. The whole scene presents a relaxed dining atmosphere.。""",
            "ref_images": {0: "<CKPT_ROOT>/hamburger.jpg"},
        },
        "hamburger_cn":{
            "prompt":  """A woman eating in a fast-food restaurant. She wears clear disposable gloves and lifts a burger to her mouth with both hands; other food sits on the table in front of her, and the restaurant's ordering screen is visible in the background. The scene has a relaxed dining atmosphere.""",
            "ref_images": {0: "<CKPT_ROOT>/hamburger.jpg"},
        },

        "waterfall":{
            "prompt": """This is a natural scenery, showcasing a magnificent waterfall landscape. The water of the waterfall cascades down from a height, creating white splashes. Surrounding it are dense forests and continuous mountains, with a clear - blue sky, presenting a beautiful scene.""",
            "ref_images": {0: "<CKPT_ROOT>/waterfall.jpg"},
        },

        # the water from the waterfall hits the rocks, creating splashes of water.
    }

    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  

    try:
        run_inference_pipeline(
            checkpoint_dirs=CHECKPOINT_DIRS,
            prompt_configs=PROMPT_CONFIGS,
            gpu_ids=GPU_IDS,
            checkpoint_interval=1300
        )
    except Exception as e:
        print(f"Main program error: {str(e)}")
        raise