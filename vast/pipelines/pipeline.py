import os
from importlib import import_module

from torch.hub import get_dir

from ..models import utils


class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LazyPipeline(BasePipeline):
    def __init__(self, pipeline, pipeline_info):
        self.pipeline = pipeline
        self.pipeline_info = pipeline_info
        self.device = None
        self.is_init = False

    def init_pipeline(self):
        if not self.is_init:
            self.pipeline = self.pipeline(**self.pipeline_info)
            if self.device is not None:
                self.pipeline.to(self.device)
            self.is_init = True

    def to(self, device):
        self.device = device
        if self.is_init:
            self.pipeline.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        return self.pipeline(*args, **kwargs)


def get_text_pipelines():
    model_dir = utils.get_model_dir()
    torch_model_dir = os.path.join(get_dir(), "checkpoints")
    pipelines = {}
    return pipelines


def get_vision_pipelines():
    model_dir = utils.get_model_dir()
    pipelines = {
        "keypoints/dwpose/yolox_l_coco_body_hand_face": {
            "_class_name": "vision.keypoints.pipeline_dwpose.DWposePipeline",
            "det_config": os.path.join(
                model_dir, "others/dwposes/yolox_l_8xb8-300e_coco.py"
            ),
            "det_ckpt": os.path.join(
                model_dir,
                "others/dwposes/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
            ),
            "pose_config": os.path.join(
                model_dir, "others/dwposes/dwpose-l_384x288.py"
            ),
            "pose_ckpt": os.path.join(model_dir, "others/dwposes/dw-ll_ucoco_384.pth"),
        },
        "keypoints/openpose/body_hand_face": {
            "_class_name": "vision.keypoints.pipeline_openpose.OpenPosePipeline",
            "model_dir": os.path.join(model_dir, "huggingface"),
            "filename": "models--lllyasviel--ControlNet/annotator/ckpts/body_pose_model.pth",
            "hand_filename": "models--lllyasviel--ControlNet/annotator/ckpts/hand_pose_model.pth",
            "face_filename": "models--lllyasviel--Annotators/facenet.pth",
        },
    }
    return pipelines


def get_pipelines():
    pipelines_list = [get_text_pipelines(), get_vision_pipelines()]
    pipelines = {}
    for pipelines_i in pipelines_list:
        for key in pipelines_i:
            assert key not in pipelines
        pipelines.update(pipelines_i)
    return pipelines


def load_pipeline(pipeline_name, lazy=False, **kwargs):
    pipelines = get_pipelines()
    pipeline_info = pipelines[pipeline_name]
    pipeline_info.update(kwargs)
    parts = pipeline_info.pop("_class_name").split(".")
    module_name = ".".join(parts[:-1])
    module = import_module("vast.pipelines." + module_name)
    pipeline = getattr(module, parts[-1])
    if lazy:
        return LazyPipeline(pipeline, pipeline_info)
    else:
        return pipeline(**pipeline_info)


def list_pipelines():
    pipelines = get_pipelines()
    return list(pipelines.keys())
