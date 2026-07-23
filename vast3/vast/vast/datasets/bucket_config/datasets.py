

from vast.datasets.datasets.clip_dataset import ClipDataset

class VideoTextDataset(ClipDataset):
    def __init__(self,
                 **kwargs,):
        super().__init__(**kwargs)


    def __getitem__(self, idx: str) -> dict:
        return (idx)    



