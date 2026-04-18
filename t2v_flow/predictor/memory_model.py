import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Any
import queue
import threading
import traceback
class DitMemoryPredictor:
    """
    一个数据驱动的DiT内存预测器。
    它通过加载OOM阈值文件和预定义的模型SP列表来动态确定可用的SP。
    """
    def __init__(self, oom_files_dir: str, model_sp_map: Dict[str, List[int]]):
        """
        初始化预测器。
        :param oom_files_dir: 存放所有 oom_thresholds_...json 文件的目录。
        :param model_sp_map: 一个字典，映射模型名称到其支持的SP列表。
        """
        self.oom_data = defaultdict(lambda: defaultdict(dict))
        
        # --- [核心修改 1] ---
        # 直接使用传入的、预先定义好的模型SP映射
        if not model_sp_map:
            raise ValueError("model_sp_map 不能为空。")
        self.model_sp_map = model_sp_map
        # --- 修改结束 ---

        self._load_oom_thresholds(oom_files_dir)
        
        if not self.oom_data:
            print("⚠️ 警告: 未能加载任何OOM阈值文件。内存预测器将无法工作。")
        else:
            print(f"✅ DitMemoryPredictor 初始化成功，已加载 {len(self.oom_data)} 个模型的OOM数据。")
            print(f"   -> 已配置的模型SP列表: {self.model_sp_map}")

    def _load_oom_thresholds(self, directory: str):
        """扫描目录，解析文件名，并加载JSON数据。"""
        print(f"   -> 正在从 '{directory}' 扫描OOM阈值文件...")
        if not os.path.isdir(directory):
            print(f"   -> ❌ 错误: OOM阈值目录不存在: {directory}")
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
                    print(f"   -> ❗️ 加载文件 '{filename}' 失败: {e}")

    @staticmethod
    def _get_resolution_key(h: int, w: int) -> str:
        """根据高和宽映射到分辨率键。"""
        if (h == 1280 and w == 720) or (h == 720 and w == 1280):
            return "720p"
        if (h == 1920 and w == 1080) or (h == 1080 and w == 1920) or (h == 1072 and w == 1936):
            return "1080p"
        if (h == 352 and w == 656):
            return "360p"
        return f"{h}x{w}"

    def get_available_sp_list(self, model_name: str, bs: int, f: int, h: int, w: int) -> List[int]:
        """
        获取给定配置下所有可用的SP值列表。
        """
        # --- [核心修改 2] ---
        # 1. 获取特定于此模型的SP列表
        if model_name not in self.model_sp_map:
            print(f"⚠️ 警告: 在配置中未找到模型 '{model_name}' 的SP列表。返回空列表。")
            return []
        
        model_specific_sps = self.model_sp_map[model_name]
        resolution_key = self._get_resolution_key(h, w)
        # print(f"model_sps = {model_specific_sps}, resolution_key = {resolution_key}")
        if model_name not in self.oom_data or resolution_key not in self.oom_data[model_name]:
            print(f"⚠️ 警告: 未找到模型 '{model_name}' 在分辨率 '{resolution_key}' 下的OOM数据。返回该模型所有可用SP。")
            return model_specific_sps

        model_res_data = self.oom_data[model_name][resolution_key]
        min_required_sp = -1

        # 2. 遍历此模型专属的SP列表
        for sp in model_specific_sps:
            # --- [核心修改] ---
            # 默认当前sp不会OOM
            is_oom = False 
            
            # 只有在OOM数据中明确找到记录时，才需要检查它是否会OOM
            if sp in model_res_data:
                sp_thresholds = model_res_data[sp]
                bs_key = str(bs)
                if bs_key in sp_thresholds:
                    max_frames = sp_thresholds[bs_key]
                    # 如果请求的帧数超过了记录的最大帧数，才标记为OOM
                    if f > max_frames:
                        is_oom = True
            
            # 如果经过检查后，is_oom标志位仍然是False，说明这个sp是可用的
            if not is_oom:
                min_required_sp = sp
                break # 找到了最小的可用SP，可以停止搜索了
            # --- 修改结束 ---

        if min_required_sp == -1:
            # print(f"return []=====================")
            return []
        
        # 3. 返回从最小可用SP开始的、该模型专属的SP列表子集
        start_index = model_specific_sps.index(min_required_sp)
        return model_specific_sps[start_index:]
        # --- 修改结束 ---
