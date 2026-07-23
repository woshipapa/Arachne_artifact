from vast.utils.prompt import clean_prompt
import random
import logging
from teleai_data_tool.schema.clip import Clip
from vast.models import utils as gm_utils
from .text_encoder import PromptEncoder
from .clip_transform import CLIPTextTransform

logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(
        self,
        # short_prompt_prob=0.5,
        short_prompt_prob=0.0,
        default_prompt="",
        # default_prompt_prob=0.2,
        default_prompt_prob=0.0,
        clean_prompt=False,
    ) -> None:
        self.short_prompt_prob = short_prompt_prob
        self.default_prompt = default_prompt
        self.default_prompt_prob = default_prompt_prob
        self.clean_prompt = clean_prompt

    def __call__(self, data_dict):
        if random.random() < self.default_prompt_prob:
            prompt = self.default_prompt
        else:
            clip: Clip = data_dict["clip_info"]
            if random.random() < self.short_prompt_prob:
                prompt = clip.caption.short_caption
            else:
                prompt = clip.caption.dense_caption

        if isinstance(prompt, list):
            prompt = random.choice(prompt)
        if self.clean_prompt:
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        data_dict["prompt"] = prompt
        return data_dict


class PromptToClipEmbedding:
    def __init__(self, model_path, dtype=None) -> None:
        self.clip_transform = CLIPTextTransform(
            gm_utils.get_model_path(model_path), dtype=dtype,
             device="cpu"
        )

    def __call__(self, data_dict):
        prompt = data_dict["prompt"]
        clip_text_embed = self.clip_transform(
            prompt, mode="after_pool", to_numpy=False
        )[0]

        data_dict["clip_text_embed"] = clip_text_embed
        return data_dict


class PromptToTransformerEmbedding:
    """
    extract text embedding from prompts
    """

    def __init__(
        self,
        model_name,
        model_path,
        max_length=None,
        with_attention_mask=False,
        padding="max_length",
    ):
        self.prompt_encoder = PromptEncoder(
            model_name, gm_utils.get_model_path(model_path), device="cpu",
        )
        self.max_length = max_length
        self.with_attention_mask = with_attention_mask
        self.padding = padding

    def __call__(self, data_dict):
        prompt = data_dict["prompt"]
        prompt_embeds, prompt_masks = self.prompt_encoder(
            prompt,
            max_length=self.max_length,
            with_attention_mask=self.with_attention_mask,
            padding=self.padding,
        )
        data_dict["prompt_embeds"] = prompt_embeds[0]
        if prompt_masks is not None:
            data_dict["prompt_masks"] = prompt_masks[0]
        return data_dict
