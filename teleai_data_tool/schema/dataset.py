from dataclasses import dataclass, field
from typing import List, Dict
from .clip import Clip, ImageWithCaption


@dataclass
class ClipsDataset:
    clips: List[Clip] = field(default_factory=list)
    clip_data_type: str = "lmdb"
    clip_data_root: str = ""
    type: str = "ClipsDataset"
    meta: Dict[str, str] = field(default_factory=dict)  # other meta info


@dataclass
class ImageDataset:
    images: List[ImageWithCaption]
    image_data_type: str = "lmdb"
    image_data_root: str = ""
    type: str = "ImageDataset"
