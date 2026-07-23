import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Any
import queue
import threading
import traceback
class DitMemoryPredictor:
    def __init__(self, oom_files_dir: str, model_sp_map: Dict[str, List[int]]):
        self.oom_data = defaultdict(lambda: defaultdict(dict))
        
        if not model_sp_map:
            raise ValueError("model_sp_map must not be empty.")
        self.model_sp_map = model_sp_map

        self._load_oom_thresholds(oom_files_dir)
        
        if not self.oom_data:
            print("[Warning] no OOM threshold file could be loaded; the memory predictor will not work.")
        else:
            print(f"[OK] DitMemoryPredictor initialised with OOM data for {len(self.oom_data)} model(s).")
            print(f"   -> configured model SP lists: {self.model_sp_map}")

    def _load_oom_thresholds(self, directory: str):
        print(f"   -> scanning '{directory}' for OOM threshold files...")
        if not os.path.isdir(directory):
            print(f"   -> [ERROR] OOM threshold directory does not exist: {directory}")
            return

        pattern = re.compile(r"oom_thresholds_(?P<model>\w+)_(?P<res>\w+)_sp(?P<sp>\d+)\.json")
        
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                data = match.groupdict()
                model_name, resolution, sp = data['model'], data['res'], int(data['sp'])
                try:
                    with open(os.path.join(directory, filename), 'r') as f:
                        self.oom_data[model_name][resolution][sp] = json.load(f)
                except Exception as e:
                    print(f"   -> failed to load '{filename}': {e}")

    @staticmethod
    def _get_resolution_key(h: int, w: int) -> str:
        if (h == 1280 and w == 720) or (h == 720 and w == 1280):
            return "720p"
        if (h == 1920 and w == 1080) or (h == 1080 and w == 1920) or (h == 1072 and w == 1936):
            return "1080p"
        if (h == 352 and w == 656):
            return "360p"
        return f"{h}x{w}"

    def get_available_sp_list(self, model_name: str, bs: int, f: int, h: int, w: int) -> List[int]:
        if model_name not in self.model_sp_map:
            print(f"[Warning] no SP list configured for model '{model_name}'. Returning an empty list.")
            return []
        
        model_specific_sps = self.model_sp_map[model_name]
        resolution_key = self._get_resolution_key(h, w)
        # print(f"model_sps = {model_specific_sps}, resolution_key = {resolution_key}")
        if model_name not in self.oom_data or resolution_key not in self.oom_data[model_name]:
            print(f"[Warning] no OOM data for model '{model_name}' at resolution '{resolution_key}'. Returning every available SP for it.")
            return model_specific_sps

        model_res_data = self.oom_data[model_name][resolution_key]
        min_required_sp = -1

        for sp in model_specific_sps:
            is_oom = False 
            
            if sp in model_res_data:
                sp_thresholds = model_res_data[sp]
                bs_key = str(bs)
                if bs_key in sp_thresholds:
                    max_frames = sp_thresholds[bs_key]
                    if f > max_frames:
                        is_oom = True
            
            if not is_oom:
                min_required_sp = sp
                break

        if min_required_sp == -1:
            # print(f"return []=====================")
            return []
        
        start_index = model_specific_sps.index(min_required_sp)
        return model_specific_sps[start_index:]
