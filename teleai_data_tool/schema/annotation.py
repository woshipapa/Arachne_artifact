from dataclasses import dataclass, field, InitVar
from typing import Optional, List, Dict
from ..utils.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh


@dataclass
class Caption:
    short_caption: List[str] = field(
        default_factory=list
    )  # the main content of the scene
    dense_caption: List[str] = field(
        default_factory=list
    )
    background: List[str] = field(default_factory=list)
    style: List[str] = field(
        default_factory=list
    )  # such as documentary, cinematic, realistic or sci-fi
    shot_type: List[str] = field(
        default_factory=list  # such as aeria shot, close-up shot, medium shot, long shot
    )
    lighting: List[str] = field(default_factory=list)
    atomophere: InitVar[list] = None
    atmosphere: List[str] = field(
        default_factory=list
    )  # such as cozy, tense, mysterious
    frame_range: List[str] = field(
        default_factory=list
    )  # the start and end frame id of this caption

    def __post_init__(self, atomophere):
        if atomophere is not None:
            self.atmosphere = atomophere


@dataclass
class FilterState:
    aesthetic: Optional[float] = None
    clearity: Optional[float] = None
    laplacian: Optional[float] = None
    optical_flow: Optional[float] = None
    ocr_score: Optional[float] = None
    water_mark: Optional[float] = None
    borders: Optional[List] = None
    logos: Optional[float] = None
    motion: Optional[float] = None
    clearity: Optional[float] = None
    video_training_suitability: Optional[float] = None
    area: Optional[float] = None
    nsfw: Optional[float] = None


@dataclass
class CameraMeta:
    shutter_speed: Optional[float] = None
    aperture: Optional[str] = None
    iso_sensitivity: Optional[int] = None
    color_space: Optional[str] = None
    others: Dict[str, str] = field(default_factory=dict)  # other meta info


@dataclass
class Instance:
    id: Optional[int] = None
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    track_id: Optional[int] = None
    bbox_type: str = "xywh"  # xywh or xyxy
    bbox: List[float] = field(default_factory=list)  # [4,]
    keypoints: Dict[str, list] = field(default_factory=dict)
    # refer to https://github.com/lidatong/dataclasses-json/issues/232
    meta: Dict[str, str] = field(default_factory=dict)  # other meta info

    def change_bbox_type(self, bbox_type: str):
        assert bbox_type in ["xyxy", "xywh"], f"{bbox_type} is not supported"
        if self.bbox_type != bbox_type:
            if self.bbox_type == "xyxy" and bbox_type == "xywh":
                self.bbox = bbox_xyxy2xywh(self.bbox)
            else:
                self.bbox = bbox_xywh2xyxy(self.bbox)
            self.bbox_type = bbox_type
